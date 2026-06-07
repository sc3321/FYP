#!/usr/bin/env python3
"""
lc_plots.py

Generate the five figures referenced in Chapter 7 from the CSV outputs
and the raw per-replicate lc_client.jsonl files.

Inputs:
    --os-per-rep-csv      Path to lc_syscall_compare_per_rep.csv (from
                          running the OS-level analysis on the OS matrix).
    --os-replicates-root  Path to the OS-level matrix root (the run that
                          had eBPF tracing attached). Used to locate the
                          bench-jsonl if not specified explicitly.
    --app-replicates-root Path to the application-level matrix root (the
                          run without eBPF tracing). Used for the
                          application-layer per-replicate p99 plots so
                          that the application numbers are not
                          contaminated by tracer overhead. If omitted,
                          falls back to --os-replicates-root.
    --bench-jsonl         Optional explicit path to one ebpf_events.jsonl
                          for the BE nanosleep histogram. Should come
                          from a cap-side case (e.g. caseD) so the
                          policy band is visible. If omitted, the script
                          looks for rep01/caseD_lc_be_long_policy/ inside
                          --os-replicates-root.
    --out-dir             Directory to write PNGs to (created if absent).

Outputs (in --out-dir):
    be_nanosleep_cluster.png       Figure 1 in Ch 7
    jk_futex_p99_strip.png         Figure 2
    jk_reshape_p95_p99.png         Figure 3
    jk_lc_client_p99_strip.png     Figure 4
    cross_layer_comparison.png     Figure 5

Each plot function is independent; if input is missing for one figure,
the others still produce.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLOR_NONE  = '#4a4a4a'
COLOR_CAP   = '#a0a0a0'
MARKER_NONE = 'o'
MARKER_CAP  = 's'

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    k = (n - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])

def read_csv_rows(path):
    p = Path(path)
    if not p.exists():
        print(f"[warn] missing csv: {path}", file=sys.stderr)
        return []
    with open(p, 'r', newline='') as fh:
        return list(csv.DictReader(fh))

def safe_float(x):
    if x is None or x == '' or x == 'None':
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None

def os_per_rep_for(rows, comparison, kind, cohort, thread_role='app'):
    out = []
    for r in rows:
        if r['comparison']  != comparison:  continue
        if r['kind']        != kind:        continue
        if r['cohort']      != cohort:      continue
        if r['thread_role'] != thread_role: continue
        out.append((
            r['rep'],
            safe_float(r.get('none_p99_us')),
            safe_float(r.get('cap_p99_us')),
            safe_float(r.get('none_p95_us')),
            safe_float(r.get('cap_p95_us')),
        ))
    out.sort(key=lambda x: x[0])
    return out

def app_per_rep_lc_client_p99(replicates_root, case_dirname, kind='lc'):
    """
    For each repNN/<case_dirname>/lc_client.jsonl, compute the p99 of
    latency_ms across all requests of the given kind. Returns
    list of (rep_label, p99).
    """
    root = Path(replicates_root)
    if not root.exists():
        print(f"[warn] missing replicates root: {root}", file=sys.stderr)
        return []
    rep_dirs = sorted(
        [d for d in root.iterdir()
         if d.is_dir() and re.match(r'^(rep|run_)\d+$', d.name)],
        key=lambda d: d.name)
    out = []
    for rep_dir in rep_dirs:
        client_path = rep_dir / case_dirname / 'lc_client.jsonl'
        if not client_path.exists():
            continue
        latencies = []
        with open(client_path, 'r') as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if kind and ev.get('kind') != kind:
                    continue
                lat = ev.get('latency_ms')
                if lat is not None:
                    latencies.append(float(lat))
        if latencies:
            out.append((rep_dir.name, percentile(latencies, 99)))
    return out

def find_default_bench_jsonl(os_replicates_root):
    """Look for rep01/caseD_*/ebpf_events.jsonl inside os_replicates_root."""
    if not os_replicates_root:
        return None
    root = Path(os_replicates_root)
    if not root.exists():
        return None
    rep_dirs = sorted(
        [d for d in root.iterdir()
         if d.is_dir() and re.match(r'^(rep|run_)\d+$', d.name)],
        key=lambda d: d.name)
    for rep_dir in rep_dirs:
        # Prefer caseD (BE-first cap) since the cluster is clearest there.
        for case_pattern in ('caseD_lc_be_long_policy',
                             'caseK_lc_cont_be_long_policy',
                             'caseH_lc_first_be_long_policy',
                             'caseF_lc_be_short_policy'):
            jsonl = rep_dir / case_pattern / 'ebpf_events.jsonl'
            if jsonl.exists():
                return jsonl
    return None

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def fig_be_nanosleep_cluster(bench_jsonl, out_dir):
    if not bench_jsonl:
        print("[skip] fig 1: no bench-jsonl found", file=sys.stderr)
        return
    p = Path(bench_jsonl)
    if not p.exists():
        print(f"[skip] fig 1: missing {p}", file=sys.stderr)
        return

    durations_ms = []
    with open(p, 'r') as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get('kind') not in ('clock_nanosleep', 'nanosleep'):
                continue
            dur_ns = ev.get('dur_ns')
            if dur_ns is None and 'dur_us' in ev:
                dur_ns = ev['dur_us'] * 1000.0
            if dur_ns is None:
                continue
            durations_ms.append(dur_ns / 1e6)

    if not durations_ms:
        print("[skip] fig 1: no nanosleep events", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    in_range = [d for d in durations_ms if d <= 20.0]
    bins = np.linspace(0, 20, 81)
    ax.hist(in_range, bins=bins, color=COLOR_NONE, edgecolor='black',
            linewidth=0.3)
    ax.axvspan(4, 8, alpha=0.12, color='red',
               label=r'policy backoff band (4--8 ms)')
    ax.set_xlabel(r'clock\_nanosleep duration (ms)')
    ax.set_ylabel('count')
    ax.set_xlim(0, 20)
    ax.legend(loc='upper right', frameon=False)
    out = Path(out_dir) / 'be_nanosleep_cluster.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"[ok] fig 1: wrote {out}", file=sys.stderr)

def fig_jk_futex_p99_strip(os_rows, out_dir):
    pairs = os_per_rep_for(os_rows,
                           comparison='lc_cont continuous (J/K)',
                           kind='futex', cohort='woken', thread_role='app')
    if not pairs:
        print("[skip] fig 2: no matching rows in OS CSV", file=sys.stderr)
        return

    none_vals = [p[1] / 1e6 for p in pairs if p[1] is not None]
    cap_vals  = [p[2] / 1e6 for p in pairs if p[2] is not None]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    rng = np.random.default_rng(0)
    x_none = rng.normal(0, 0.04, size=len(none_vals))
    x_cap  = rng.normal(1, 0.04, size=len(cap_vals))

    ax.scatter(x_none, none_vals, color=COLOR_NONE, marker=MARKER_NONE,
               s=28, label=f'case J (none), $n={len(none_vals)}$',
               edgecolor='black', linewidth=0.3)
    ax.scatter(x_cap, cap_vals, color=COLOR_CAP, marker=MARKER_CAP,
               s=28, label=f'case K (cap), $n={len(cap_vals)}$',
               edgecolor='black', linewidth=0.3)
    ax.plot([-0.25, 0.25], [np.mean(none_vals)] * 2,
            color=COLOR_NONE, linewidth=2)
    ax.plot([0.75, 1.25], [np.mean(cap_vals)] * 2,
            color='black', linewidth=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['none (case J)', 'cap (case K)'])
    ax.set_ylabel(r'LC futex wake-cohort p99 (s)')
    ax.set_xlim(-0.6, 1.6)
    ax.legend(loc='upper right', frameon=False)
    delta_pct = (np.mean(cap_vals) - np.mean(none_vals)) / np.mean(none_vals) * 100
    ax.text(0.5, ax.get_ylim()[1] * 0.97,
            f'mean $\\Delta = {delta_pct:+.1f}\\%$',
            ha='center', va='top', fontsize=10)
    out = Path(out_dir) / 'jk_futex_p99_strip.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"[ok] fig 2: wrote {out}", file=sys.stderr)

def fig_jk_reshape_p95_p99(os_rows, out_dir):
    pairs = os_per_rep_for(os_rows,
                           comparison='lc_cont continuous (J/K)',
                           kind='futex', cohort='woken', thread_role='app')
    if not pairs:
        print("[skip] fig 3: no matching rows in OS CSV", file=sys.stderr)
        return

    none_p95 = [p[3] for p in pairs if p[3] is not None]
    cap_p95  = [p[4] for p in pairs if p[4] is not None]
    none_p99 = [p[1] / 1e6 for p in pairs if p[1] is not None]
    cap_p99  = [p[2] / 1e6 for p in pairs if p[2] is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5))
    rng = np.random.default_rng(1)

    x1n = rng.normal(0, 0.04, size=len(none_p95))
    x1c = rng.normal(1, 0.04, size=len(cap_p95))
    ax1.scatter(x1n, none_p95, color=COLOR_NONE, marker=MARKER_NONE,
                s=24, edgecolor='black', linewidth=0.3)
    ax1.scatter(x1c, cap_p95, color=COLOR_CAP, marker=MARKER_CAP,
                s=24, edgecolor='black', linewidth=0.3)
    ax1.plot([-0.25, 0.25], [np.mean(none_p95)] * 2,
             color=COLOR_NONE, linewidth=2)
    ax1.plot([0.75, 1.25], [np.mean(cap_p95)] * 2,
             color='black', linewidth=2)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['none', 'cap'])
    ax1.set_ylabel(r'p95 ($\mu$s)')
    ax1.set_title('95th percentile (body)')
    ax1.set_xlim(-0.6, 1.6)
    d95 = (np.mean(cap_p95) - np.mean(none_p95)) / np.mean(none_p95) * 100
    ax1.text(0.5, ax1.get_ylim()[1] * 0.97,
             f'$\\Delta = {d95:+.1f}\\%$',
             ha='center', va='top', fontsize=10)

    x2n = rng.normal(0, 0.04, size=len(none_p99))
    x2c = rng.normal(1, 0.04, size=len(cap_p99))
    ax2.scatter(x2n, none_p99, color=COLOR_NONE, marker=MARKER_NONE,
                s=24, edgecolor='black', linewidth=0.3)
    ax2.scatter(x2c, cap_p99, color=COLOR_CAP, marker=MARKER_CAP,
                s=24, edgecolor='black', linewidth=0.3)
    ax2.plot([-0.25, 0.25], [np.mean(none_p99)] * 2,
             color=COLOR_NONE, linewidth=2)
    ax2.plot([0.75, 1.25], [np.mean(cap_p99)] * 2,
             color='black', linewidth=2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['none', 'cap'])
    ax2.set_ylabel('p99 (s)')
    ax2.set_title('99th percentile (tail)')
    ax2.set_xlim(-0.6, 1.6)
    d99 = (np.mean(cap_p99) - np.mean(none_p99)) / np.mean(none_p99) * 100
    ax2.text(0.5, ax2.get_ylim()[1] * 0.97,
             f'$\\Delta = {d99:+.1f}\\%$',
             ha='center', va='top', fontsize=10)

    fig.suptitle('Case J vs K: LC futex wake-cohort distribution reshape',
                 y=1.02)
    out = Path(out_dir) / 'jk_reshape_p95_p99.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"[ok] fig 3: wrote {out}", file=sys.stderr)

def fig_jk_lc_client_p99_strip(app_root, out_dir):
    if not app_root:
        print("[skip] fig 4: no app-replicates-root", file=sys.stderr)
        return

    j_pairs = app_per_rep_lc_client_p99(app_root,
                                        'caseJ_lc_cont_be_long_none',
                                        kind='lc')
    k_pairs = app_per_rep_lc_client_p99(app_root,
                                        'caseK_lc_cont_be_long_policy',
                                        kind='lc')
    j_vals = [v for _, v in j_pairs if v is not None]
    k_vals = [v for _, v in k_pairs if v is not None]

    if not j_vals or not k_vals:
        print(f"[skip] fig 4: no values "
              f"(j={len(j_vals)}, k={len(k_vals)})", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    rng = np.random.default_rng(2)
    x_j = rng.normal(0, 0.04, size=len(j_vals))
    x_k = rng.normal(1, 0.04, size=len(k_vals))

    ax.scatter(x_j, j_vals, color=COLOR_NONE, marker=MARKER_NONE, s=28,
               label=f'case J (none), $n={len(j_vals)}$',
               edgecolor='black', linewidth=0.3)
    ax.scatter(x_k, k_vals, color=COLOR_CAP, marker=MARKER_CAP, s=28,
               label=f'case K (cap), $n={len(k_vals)}$',
               edgecolor='black', linewidth=0.3)
    ax.plot([-0.25, 0.25], [np.mean(j_vals)] * 2,
            color=COLOR_NONE, linewidth=2)
    ax.plot([0.75, 1.25], [np.mean(k_vals)] * 2,
            color='black', linewidth=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['none (case J)', 'cap (case K)'])
    ax.set_ylabel('LC client p99 (ms)')
    ax.set_xlim(-0.6, 1.6)
    ax.legend(loc='upper right', frameon=False)
    delta_pct = (np.mean(k_vals) - np.mean(j_vals)) / np.mean(j_vals) * 100
    ax.text(0.5, ax.get_ylim()[1] * 0.97,
            f'mean $\\Delta = {delta_pct:+.1f}\\%$',
            ha='center', va='top', fontsize=10)
    out = Path(out_dir) / 'jk_lc_client_p99_strip.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"[ok] fig 4: wrote {out}", file=sys.stderr)

def fig_cross_layer_comparison(os_rows, app_root, out_dir):
    def os_deltas(comparison):
        return [
            safe_float(r['p99_delta_pct'])
            for r in os_rows
            if r['comparison']  == comparison
            and r['kind']       == 'futex'
            and r['cohort']     == 'woken'
            and r['thread_role'] == 'app'
            and safe_float(r['p99_delta_pct']) is not None
        ]

    os_cd = os_deltas('be_first sequential (C/D)')
    os_ef = os_deltas('be_short neg-control (E/F)')
    os_jk = os_deltas('lc_cont continuous (J/K)')

    def app_deltas(case_none, case_cap):
        if not app_root:
            return []
        none_pairs = dict(app_per_rep_lc_client_p99(app_root, case_none, kind='lc'))
        cap_pairs  = dict(app_per_rep_lc_client_p99(app_root, case_cap,  kind='lc'))
        out = []
        for rep, none_p99 in none_pairs.items():
            cap_p99 = cap_pairs.get(rep)
            if none_p99 and cap_p99:
                out.append((cap_p99 - none_p99) / none_p99 * 100)
        return out

    app_cd = app_deltas('caseC_lc_be_long_none',
                        'caseD_lc_be_long_policy')
    app_ef = app_deltas('caseE_lc_be_short_none',
                        'caseF_lc_be_short_policy')
    app_jk = app_deltas('caseJ_lc_cont_be_long_none',
                        'caseK_lc_cont_be_long_policy')

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    positions_os  = [0.0, 1.5, 3.0]
    positions_app = [0.6, 2.1, 3.6]
    data_os  = [os_cd, os_ef, os_jk]
    data_app = [app_cd, app_ef, app_jk]

    bp_os = ax.boxplot(data_os, positions=positions_os, widths=0.45,
                       patch_artist=True, showfliers=True,
                       medianprops={'color': 'black', 'linewidth': 1.3})
    bp_app = ax.boxplot(data_app, positions=positions_app, widths=0.45,
                        patch_artist=True, showfliers=True,
                        medianprops={'color': 'black', 'linewidth': 1.3})

    for patch in bp_os['boxes']:
        patch.set_facecolor(COLOR_NONE)
        patch.set_alpha(0.6)
    for patch in bp_app['boxes']:
        patch.set_facecolor(COLOR_CAP)
        patch.set_alpha(0.85)

    ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
    ax.set_xticks([0.3, 1.8, 3.3])
    ax.set_xticklabels(['C/D\nsequential',
                        'E/F\nneg control',
                        'J/K\ncontinuous'])
    ax.set_ylabel(r'p99 $\Delta$ (cap $-$ none), \%')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_NONE, alpha=0.6,
              label='OS layer (futex wake-cohort p99)'),
        Patch(facecolor=COLOR_CAP, alpha=0.85,
              label='App layer (LC client p99)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', frameon=False)

    out = Path(out_dir) / 'cross_layer_comparison.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"[ok] fig 5: wrote {out}", file=sys.stderr)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--os-per-rep-csv', required=True,
                    help='Path to lc_syscall_compare_per_rep.csv (OS-level).')
    ap.add_argument('--os-replicates-root', required=True,
                    help='OS-level matrix root (the one with eBPF tracing).')
    ap.add_argument('--app-replicates-root', default=None,
                    help='Application-level matrix root (the one without '
                         'eBPF tracing). Falls back to --os-replicates-root '
                         'if not given.')
    ap.add_argument('--bench-jsonl', default=None,
                    help='Optional explicit ebpf_events.jsonl for the BE '
                         'nanosleep histogram. If omitted, auto-found inside '
                         '--os-replicates-root.')
    ap.add_argument('--out-dir', default='./figures')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os_rows = read_csv_rows(args.os_per_rep_csv)
    if not os_rows:
        print(f"[error] no rows in {args.os_per_rep_csv}", file=sys.stderr)
        sys.exit(1)

    app_root = args.app_replicates_root or args.os_replicates_root
    if app_root == args.os_replicates_root:
        print("[note] --app-replicates-root not given; using OS root for "
              "application-layer plots (these will include tracer overhead).",
              file=sys.stderr)

    bench = args.bench_jsonl or find_default_bench_jsonl(args.os_replicates_root)

    fig_be_nanosleep_cluster(bench, out_dir)
    fig_jk_futex_p99_strip(os_rows, out_dir)
    fig_jk_reshape_p95_p99(os_rows, out_dir)
    fig_jk_lc_client_p99_strip(app_root, out_dir)
    fig_cross_layer_comparison(os_rows, app_root, out_dir)

    print(f"\nFigures written to: {out_dir}", file=sys.stderr)

if __name__ == '__main__':
    main()
