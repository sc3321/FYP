#!/usr/bin/env python3
"""
Correlation analysis: instrumented (manual API) vs vanilla+LD_PRELOAD event streams.

What this script answers
------------------------
The §6.6 / §8.3 claim is that AUTO_CUDA_* events from the unmodified binary
exhibit the same structural shape as LLAMA_DECODE events from the instrumented
binary, on the same workload. This script quantifies that.

The two arms run sequentially with server restarts, so they do not share a
clock origin. Comparison is therefore within-request shape, not absolute time:

  1.  Detect request boundaries in each arm.
      - instr:    LLAMA_REQUEST phases mark them directly.
      - preload:  gap detection on AUTO_CUDA event activity.
  2.  For each request, normalise time to [0, 1] and bin AUTO_CUDA /
      LLAMA_DECODE events into the same number of bins.
  3.  Average across requests to get a canonical event-density curve per arm.
  4.  Pearson r between the two canonical curves is the headline number.

Outputs (under <run>/analysis/):
  summary.csv             per-rep aggregate statistics and correlation
  timeline_repNN.png      side-by-side timeline of one rep, both arms
  canonical_shape.png     averaged event density vs normalised time
  per_request_counts.png  scatter: instr decode count vs preload sync count
  inter_event_dist.png    inter-event interval distribution per arm

Usage
-----
  python analyze_correlation.py /path/to/runs/correlation_TIMESTAMP
  python analyze_correlation.py /path/to/runs/correlation_TIMESTAMP --latest

If the run directory contains a single rep, only the per-rep plots are produced.
"""

import argparse
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
# Sample line, as emitted by eventHandler::writeEvent:
#   Event type = BEGIN: PhaseId:[12345, 7],Thread Id: 12346, parent_id: 0,
#   depth: 0,  Timestamp: 5837 s 412938273 ns, phase type: (LLAMA_DECODE),
#   workload class: LC
LINE_RE = re.compile(
    r'Event type = (?P<kind>BEGIN|END):\s*'
    r'PhaseId:\[(?P<pid>\d+),\s*(?P<phase_counter>\d+)\],\s*'
    r'Thread Id:\s*(?P<tid>\d+),\s*'
    r'parent_id:\s*(?P<parent_id>\d+),\s*'
    r'depth:\s*(?P<depth>\d+),\s*+'
    r'Timestamp:\s*(?P<sec>\d+)\s*s\s*(?P<ns>\d+)\s*ns,\s*'
    r'phase type:\s*\((?P<label>[^)]+)\),\s*'
    r'workload class:\s*(?P<wclass>\w+)'
)


def parse_event_file(path: Path) -> pd.DataFrame:
    """Parse one per-PID event log into a DataFrame."""
    rows = []
    with path.open('r', errors='replace') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            m = LINE_RE.match(line)
            if not m:
                # Unparseable lines are a sign of format drift. Don't silently
                # drop them; warn so the regex can be adjusted.
                if line_no <= 3 or line_no % 1000 == 0:
                    print(f"  warn: unparseable line {path.name}:{line_no}",
                          file=sys.stderr)
                continue
            d = m.groupdict()
            rows.append({
                'kind': d['kind'],
                'pid': int(d['pid']),
                'phase_counter': int(d['phase_counter']),
                'tid': int(d['tid']),
                'parent_id': int(d['parent_id']),
                'depth': int(d['depth']),
                'ts_ns': int(d['sec']) * 1_000_000_000 + int(d['ns']),
                'label': d['label'],
                'wclass': d['wclass'],
            })
    return pd.DataFrame(rows)


def load_arm_events(events_dir: Path) -> pd.DataFrame:
    """Load all per-PID event files from one arm's lc_events directory."""
    if not events_dir.is_dir():
        return pd.DataFrame()
    parts = []
    for p in sorted(events_dir.iterdir()):
        # Per-PID files are named numerically; skip stdout/stderr logs.
        if not p.is_file():
            continue
        if not p.name.isdigit():
            continue
        df = parse_event_file(p)
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df.sort_values('ts_ns', inplace=True, ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Phase reconstruction (BEGIN/END -> intervals)
# ---------------------------------------------------------------------------
def reconstruct_phases(df: pd.DataFrame) -> pd.DataFrame:
    """Pair BEGIN/END rows by (pid, phase_counter) into one row per phase."""
    if df.empty:
        return df
    df = df.copy()
    df['phase_key'] = list(zip(df['pid'], df['phase_counter']))
    begins = df[df['kind'] == 'BEGIN'].set_index('phase_key')
    ends = df[df['kind'] == 'END'].set_index('phase_key')
    common = begins.index.intersection(ends.index)
    if len(common) == 0:
        return pd.DataFrame()
    out = pd.DataFrame({
        'pid': begins.loc[common, 'pid'].values,
        'tid': begins.loc[common, 'tid'].values,
        'label': begins.loc[common, 'label'].values,
        'wclass': begins.loc[common, 'wclass'].values,
        'depth': begins.loc[common, 'depth'].values,
        'parent_id': begins.loc[common, 'parent_id'].values,
        'start_ns': begins.loc[common, 'ts_ns'].values,
        'end_ns': ends.loc[common, 'ts_ns'].values,
    })
    out['dur_ns'] = out['end_ns'] - out['start_ns']
    out.sort_values('start_ns', inplace=True, ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# Request boundary detection
# ---------------------------------------------------------------------------
@dataclass
class RequestBoundary:
    start_ns: int
    end_ns: int
    source: str   # how we identified it


def detect_requests_instr(phases: pd.DataFrame) -> list[RequestBoundary]:
    """In the instrumented arm, LLAMA_REQUEST phases mark request boundaries."""
    if phases.empty:
        return []
    req = phases[phases['label'] == 'LLAMA_REQUEST']
    return [
        RequestBoundary(int(r.start_ns), int(r.end_ns), 'LLAMA_REQUEST')
        for r in req.itertuples()
    ]


def detect_requests_preload(events: pd.DataFrame,
                            gap_threshold_ms: float = 100.0
                            ) -> list[RequestBoundary]:
    """In the preload arm there is no request label. Detect request boundaries
    from gaps in AUTO_CUDA event activity. Inside a request, sync calls are
    dense; between requests, the server is idle waiting for the next HTTP
    request, so the gap is much longer than the inter-sync interval.
    """
    if events.empty:
        return []
    auto = events[events['label'].str.startswith('AUTO_CUDA')].copy()
    if auto.empty:
        return []
    # Use only BEGIN events for boundary detection; one event per sync call
    # is enough and avoids double-counting BEGIN/END pairs.
    begins = auto[auto['kind'] == 'BEGIN'].sort_values('ts_ns')
    if begins.empty:
        return []
    ts = begins['ts_ns'].values
    gaps = np.diff(ts)
    gap_threshold_ns = int(gap_threshold_ms * 1_000_000)
    # A new request starts at the event after each large gap. Always include
    # the first event as the start of the first request.
    boundaries = [0] + (np.where(gaps > gap_threshold_ns)[0] + 1).tolist() + [len(ts)]
    out = []
    for i in range(len(boundaries) - 1):
        s = ts[boundaries[i]]
        # End of this request = last event in the cluster.
        e = ts[boundaries[i + 1] - 1]
        out.append(RequestBoundary(int(s), int(e), f'gap>{gap_threshold_ms}ms'))
    return out


# ---------------------------------------------------------------------------
# Per-request event density
# ---------------------------------------------------------------------------
def event_density_per_request(events_ts_ns: np.ndarray,
                              requests: list[RequestBoundary],
                              n_bins: int = 40,
                              drop_first_n: int = 1
                              ) -> np.ndarray:
    """For each request, bin events into n_bins normalised time bins.

    Returns a (n_requests - drop_first_n, n_bins) array of event counts.
    The first request is dropped by default since warmup effects often
    skew it (kernel JIT, page-in, allocator churn).
    """
    if not requests or len(events_ts_ns) == 0:
        return np.zeros((0, n_bins))
    requests = requests[drop_first_n:]
    rows = []
    for r in requests:
        if r.end_ns <= r.start_ns:
            continue
        mask = (events_ts_ns >= r.start_ns) & (events_ts_ns <= r.end_ns)
        ts_in = events_ts_ns[mask]
        if len(ts_in) == 0:
            rows.append(np.zeros(n_bins))
            continue
        normalised = (ts_in - r.start_ns) / (r.end_ns - r.start_ns)
        counts, _ = np.histogram(normalised, bins=n_bins, range=(0.0, 1.0))
        rows.append(counts)
    if not rows:
        return np.zeros((0, n_bins))
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Analysis per rep
# ---------------------------------------------------------------------------
@dataclass
class RepAnalysis:
    rep_name: str
    instr_events: pd.DataFrame
    preload_events: pd.DataFrame
    instr_phases: pd.DataFrame
    preload_phases: pd.DataFrame
    instr_requests: list[RequestBoundary]
    preload_requests: list[RequestBoundary]
    instr_density: np.ndarray
    preload_density: np.ndarray
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float


def analyse_rep(rep_dir: Path, n_bins: int = 40) -> RepAnalysis | None:
    instr_dir = rep_dir / 'instr' / 'lc_events'
    preload_dir = rep_dir / 'preload' / 'lc_events'
    instr_events = load_arm_events(instr_dir)
    preload_events = load_arm_events(preload_dir)
    if instr_events.empty or preload_events.empty:
        print(f"  skip {rep_dir.name}: missing events "
              f"(instr={len(instr_events)}, preload={len(preload_events)})",
              file=sys.stderr)
        return None

    instr_phases = reconstruct_phases(instr_events)
    preload_phases = reconstruct_phases(preload_events)

    instr_requests = detect_requests_instr(instr_phases)
    preload_requests = detect_requests_preload(preload_events)

    # Events we count per request:
    #   instr:    LLAMA_DECODE BEGINs (one per decode step)
    #   preload:  AUTO_CUDA_* BEGINs (one per intercepted sync call)
    instr_decode_begins = instr_events[
        (instr_events['label'] == 'LLAMA_DECODE')
        & (instr_events['kind'] == 'BEGIN')
    ]['ts_ns'].values
    preload_sync_begins = preload_events[
        (preload_events['label'].str.startswith('AUTO_CUDA'))
        & (preload_events['kind'] == 'BEGIN')
    ]['ts_ns'].values

    instr_density = event_density_per_request(
        instr_decode_begins, instr_requests, n_bins=n_bins)
    preload_density = event_density_per_request(
        preload_sync_begins, preload_requests, n_bins=n_bins)

    # Canonical shapes: mean event count per normalised bin.
    if instr_density.shape[0] == 0 or preload_density.shape[0] == 0:
        pr = pp = sr = sp = float('nan')
    else:
        instr_canon = instr_density.mean(axis=0)
        preload_canon = preload_density.mean(axis=0)
        # Pearson and Spearman on the two canonical curves.
        if instr_canon.std() == 0 or preload_canon.std() == 0:
            pr = pp = sr = sp = float('nan')
        else:
            pr, pp = pearsonr(instr_canon, preload_canon)
            sr, sp = spearmanr(instr_canon, preload_canon)

    return RepAnalysis(
        rep_name=rep_dir.name,
        instr_events=instr_events,
        preload_events=preload_events,
        instr_phases=instr_phases,
        preload_phases=preload_phases,
        instr_requests=instr_requests,
        preload_requests=preload_requests,
        instr_density=instr_density,
        preload_density=preload_density,
        pearson_r=pr,
        pearson_p=pp,
        spearman_r=sr,
        spearman_p=sp,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_timeline(rep: RepAnalysis, out_path: Path,
                  request_index: int = 2):
    """Side-by-side timeline of one representative request from each arm.

    The two arms ran sequentially so absolute time differs. Each panel uses
    its own t=0 at the request's first event. The point is the SHAPE.
    """
    instr_decode = rep.instr_phases[rep.instr_phases['label'] == 'LLAMA_DECODE']
    preload_sync = rep.preload_phases[
        rep.preload_phases['label'].str.startswith('AUTO_CUDA')
    ]

    if request_index >= len(rep.instr_requests) or \
       request_index >= len(rep.preload_requests):
        request_index = 0
    if not rep.instr_requests or not rep.preload_requests:
        return

    ir = rep.instr_requests[request_index]
    pr = rep.preload_requests[request_index]

    instr_in = instr_decode[
        (instr_decode['start_ns'] >= ir.start_ns)
        & (instr_decode['end_ns'] <= ir.end_ns)
    ].copy()
    preload_in = preload_sync[
        (preload_sync['start_ns'] >= pr.start_ns)
        & (preload_sync['end_ns'] <= pr.end_ns)
    ].copy()

    fig, axes = plt.subplots(2, 1, figsize=(11, 4), sharex=False)

    # Instr panel
    ax = axes[0]
    for r in instr_in.itertuples():
        s = (r.start_ns - ir.start_ns) / 1e6
        e = (r.end_ns - ir.start_ns) / 1e6
        ax.barh(0, e - s, left=s, height=0.6, color='#2c7fb8')
    ax.set_xlim(0, (ir.end_ns - ir.start_ns) / 1e6)
    ax.set_yticks([])
    ax.set_ylabel('instr\nLLAMA_DECODE', rotation=0, labelpad=40, va='center')
    ax.set_title(f'{rep.rep_name} — request #{request_index} timeline '
                 f'(instr top, preload bottom; independent time origins)')

    # Preload panel
    ax = axes[1]
    palette = {
        'AUTO_CUDA_STREAM_SYNC': '#e6550d',
        'AUTO_CUDA_DEV_SYNC': '#31a354',
        'AUTO_CUDA_EVENT_SYNC': '#756bb1',
    }
    for r in preload_in.itertuples():
        s = (r.start_ns - pr.start_ns) / 1e6
        e = (r.end_ns - pr.start_ns) / 1e6
        ax.barh(0, max(e - s, 0.1), left=s, height=0.6,
                color=palette.get(r.label, '#888'))
    ax.set_xlim(0, (pr.end_ns - pr.start_ns) / 1e6)
    ax.set_yticks([])
    ax.set_ylabel('preload\nAUTO_CUDA_*', rotation=0, labelpad=40, va='center')
    ax.set_xlabel('time within request (ms)')

    # Legend for preload panel
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=l)
        for l, c in palette.items()
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8, ncol=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_canonical_shape(reps: list[RepAnalysis], out_path: Path,
                          n_bins: int = 40):
    """Mean event-density curve across all requests of all reps, per arm."""
    if not reps:
        return
    instr_stack = np.vstack([r.instr_density for r in reps
                             if r.instr_density.shape[0] > 0])
    preload_stack = np.vstack([r.preload_density for r in reps
                               if r.preload_density.shape[0] > 0])
    if instr_stack.shape[0] == 0 or preload_stack.shape[0] == 0:
        return

    bin_centres = (np.arange(n_bins) + 0.5) / n_bins
    instr_mean = instr_stack.mean(axis=0)
    instr_sem = instr_stack.std(axis=0) / np.sqrt(instr_stack.shape[0])
    preload_mean = preload_stack.mean(axis=0)
    preload_sem = preload_stack.std(axis=0) / np.sqrt(preload_stack.shape[0])

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(bin_centres, instr_mean, color='#2c7fb8',
            label=f'instr LLAMA_DECODE (n={instr_stack.shape[0]} requests)')
    ax.fill_between(bin_centres,
                    instr_mean - instr_sem,
                    instr_mean + instr_sem,
                    color='#2c7fb8', alpha=0.2)

    ax.plot(bin_centres, preload_mean, color='#e6550d',
            label=f'preload AUTO_CUDA_* (n={preload_stack.shape[0]} requests)')
    ax.fill_between(bin_centres,
                    preload_mean - preload_sem,
                    preload_mean + preload_sem,
                    color='#e6550d', alpha=0.2)

    # Correlation between the two pooled canonical curves.
    if instr_mean.std() > 0 and preload_mean.std() > 0:
        pr, pp = pearsonr(instr_mean, preload_mean)
        sr, sp = spearmanr(instr_mean, preload_mean)
        ax.text(0.02, 0.97,
                f'Pearson r = {pr:.3f} (p = {pp:.2e})\n'
                f'Spearman ρ = {sr:.3f} (p = {sp:.2e})',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9))

    ax.set_xlabel('normalised time within request')
    ax.set_ylabel('mean event count per bin')
    ax.set_title('Canonical event-density shape, pooled across requests')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_per_request_counts(reps: list[RepAnalysis], out_path: Path):
    """Scatter of per-request event counts: instr decode vs preload sync.

    Strictly: request N in instr arm corresponds to request N in preload arm
    only if the workload is deterministic and matched. With model sampling
    on (default temperature > 0) the tokens generated will differ. We still
    expect a positive trend.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    all_x, all_y = [], []
    for rep in reps:
        ic = rep.instr_density.sum(axis=1)
        pc = rep.preload_density.sum(axis=1)
        n = min(len(ic), len(pc))
        if n == 0:
            continue
        ax.scatter(ic[:n], pc[:n], alpha=0.6, label=rep.rep_name, s=25)
        all_x.extend(ic[:n].tolist())
        all_y.extend(pc[:n].tolist())

    if all_x and all_y:
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        if all_x.std() > 0 and all_y.std() > 0:
            pr, pp = pearsonr(all_x, all_y)
            sr, sp = spearmanr(all_x, all_y)
            ax.text(0.02, 0.97,
                    f'n = {len(all_x)} requests\n'
                    f'Pearson r = {pr:.3f} (p = {pp:.2e})\n'
                    f'Spearman ρ = {sr:.3f} (p = {sp:.2e})',
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9))
        lo = min(all_x.min(), all_y.min())
        hi = max(all_x.max(), all_y.max())
        ax.plot([lo, hi], [lo, hi], color='gray', linestyle=':',
                alpha=0.6, label='y = x')

    ax.set_xlabel('instr: LLAMA_DECODE events per request')
    ax.set_ylabel('preload: AUTO_CUDA_* events per request')
    ax.set_title('Per-request event counts')
    if len(reps) > 1:
        ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_inter_event_dist(reps: list[RepAnalysis], out_path: Path):
    """Distribution of inter-event intervals per arm. Should be bimodal:
    short intervals = within-request submission cadence, long intervals =
    inter-request quiescence. Both arms should show the same modes if they
    are tracking the same underlying activity.
    """
    instr_ts = np.concatenate([
        r.instr_events[
            (r.instr_events['label'] == 'LLAMA_DECODE')
            & (r.instr_events['kind'] == 'BEGIN')
        ]['ts_ns'].values
        for r in reps
    ]) if reps else np.array([])
    preload_ts = np.concatenate([
        r.preload_events[
            (r.preload_events['label'].str.startswith('AUTO_CUDA'))
            & (r.preload_events['kind'] == 'BEGIN')
        ]['ts_ns'].values
        for r in reps
    ]) if reps else np.array([])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, ts, title, colour in [
        (axes[0], instr_ts, 'instr LLAMA_DECODE', '#2c7fb8'),
        (axes[1], preload_ts, 'preload AUTO_CUDA_*', '#e6550d'),
    ]:
        if len(ts) < 2:
            continue
        ts = np.sort(ts)
        gaps_ms = np.diff(ts) / 1e6
        gaps_ms = gaps_ms[gaps_ms > 0]
        ax.hist(np.log10(gaps_ms), bins=60, color=colour, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel('log10(inter-event interval, ms)')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('count')
    fig.suptitle('Inter-event intervals (log scale)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def find_reps(run_dir: Path) -> list[Path]:
    reps = sorted(p for p in run_dir.iterdir()
                  if p.is_dir() and p.name.startswith('rep'))
    return reps


def write_summary(reps: list[RepAnalysis], out_path: Path):
    rows = []
    for rep in reps:
        rows.append({
            'rep': rep.rep_name,
            'instr_events_total': len(rep.instr_events),
            'preload_events_total': len(rep.preload_events),
            'instr_requests_detected': len(rep.instr_requests),
            'preload_requests_detected': len(rep.preload_requests),
            'instr_decode_events': int(rep.instr_density.sum()),
            'preload_sync_events': int(rep.preload_density.sum()),
            'instr_decode_per_req_mean': float(rep.instr_density.sum(axis=1).mean())
                if rep.instr_density.shape[0] > 0 else float('nan'),
            'preload_sync_per_req_mean': float(rep.preload_density.sum(axis=1).mean())
                if rep.preload_density.shape[0] > 0 else float('nan'),
            'canonical_pearson_r': rep.pearson_r,
            'canonical_pearson_p': rep.pearson_p,
            'canonical_spearman_r': rep.spearman_r,
            'canonical_spearman_p': rep.spearman_p,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', type=Path,
                    help='correlation_TIMESTAMP directory produced by run_correlation.sh')
    ap.add_argument('--n-bins', type=int, default=40,
                    help='number of normalised-time bins per request (default: 40)')
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"error: {run_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    rep_dirs = find_reps(run_dir)
    if not rep_dirs:
        print(f"error: no rep* subdirectories under {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Analysing {len(rep_dirs)} replicate(s) in {run_dir}")

    out_dir = run_dir / 'analysis'
    out_dir.mkdir(exist_ok=True)

    reps: list[RepAnalysis] = []
    for rep_dir in rep_dirs:
        print(f"  {rep_dir.name} ...")
        result = analyse_rep(rep_dir, n_bins=args.n_bins)
        if result is None:
            continue
        reps.append(result)
        # Per-rep timeline plot
        plot_timeline(result, out_dir / f'timeline_{rep_dir.name}.png')

    if not reps:
        print("error: no usable replicates", file=sys.stderr)
        sys.exit(1)

    # Aggregate plots and summary
    plot_canonical_shape(reps, out_dir / 'canonical_shape.png',
                          n_bins=args.n_bins)
    plot_per_request_counts(reps, out_dir / 'per_request_counts.png')
    plot_inter_event_dist(reps, out_dir / 'inter_event_dist.png')
    summary = write_summary(reps, out_dir / 'summary.csv')

    print()
    print("Per-rep summary:")
    with pd.option_context('display.width', 140,
                           'display.max_columns', None,
                           'display.float_format', '{:.3f}'.format):
        print(summary.to_string(index=False))

    print()
    print(f"Outputs written to {out_dir}/")


if __name__ == '__main__':
    main()
