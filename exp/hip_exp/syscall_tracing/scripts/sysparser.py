#!/usr/bin/env python3
"""
sysparser.py (updated)

Parses `strace -c` summaries produced by sweep_strace.sh, matching this layout:

  results/
    tb_cpu/
      run_<timestamp>/
        alloc_<bytes>b_<iters>iters_rep1.txt
        kernels_<bytes>b_<iters>iters_rep2.txt
        ...
      latest -> run_<timestamp>
    tb_cuda/
      run_<timestamp>/
      latest -> run_<timestamp>

Key updates vs your previous parser:
  - backend dimension: inferred from path part "tb_cpu" / "tb_cuda"
  - variant inferred from filename prefix (alloc/baseline/kernels)
  - supports *_repN.txt filenames
  - aggregates repeats: mean over reps for each (backend,variant,bytes,iters,syscall)

Outputs plots per backend under:
  <out>/<backend>/<variant>/...

"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

MB = 1024 * 1024
KB = 1024

# Example filename:
#   alloc_1000000b_5000iters_rep1.txt
#   kernels_10485760b_10000iters_rep2.txt
FNAME_RE = re.compile(
    r"^(?P<variant>alloc|baseline|kernels)_(?P<bytes>\d+)b_(?P<iters>\d+)iters_(?:rep(?P<rep>\d+))\.txt$",
    re.IGNORECASE,
)

# Tolerant strace -c row regex:
ROW_RE = re.compile(
    r"""
    ^\s*
    (?P<pct>\d+(?:\.\d+)?)\s+
    (?P<sec>\d+(?:\.\d+)?)\s+
    (?P<usec>\d+)\s+
    (?P<calls>\d+)\s+
    (?:(?P<errors>\d+)\s+)?          # optional errors
    (?P<syscall>\S+)\s*
    $
    """,
    re.VERBOSE,
)

def parse_strace_c(text: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("% time"):
            continue
        if set(line) <= {"-"}:
            continue
        if line.split() and line.split()[-1] == "total":
            continue

        m = ROW_RE.match(raw) or ROW_RE.match(line)
        if not m:
            continue

        rows.append(
            {
                "syscall": m.group("syscall"),
                "pct_time": float(m.group("pct")),
                "calls": int(m.group("calls")),
                "seconds": float(m.group("sec")),
                "usecs_per_call": int(m.group("usec")),
                "errors": int(m.group("errors")) if m.group("errors") is not None else 0,
            }
        )
    return rows


@dataclass(frozen=True)
class Record:
    backend: str
    variant: str
    bytes: int
    iterations: int
    syscall: str
    pct_time: float
    calls: int


def parse_filename(fp: Path) -> Optional[Tuple[str, int, int, int]]:
    """
    Returns: (variant, bytes, iters, rep)
    """
    m = FNAME_RE.match(fp.name)
    if not m:
        return None
    variant = m.group("variant").lower()
    bytes_ = int(m.group("bytes"))
    iters = int(m.group("iters"))
    rep = int(m.group("rep")) if m.group("rep") is not None else 1
    return variant, bytes_, iters, rep


def infer_backend_from_path(fp: Path, known_backends: Optional[List[str]] = None) -> Optional[str]:
    """
    Looks for a path component like tb_cpu, tb_cuda, tb_hip.
    If known_backends provided, only returns one of those.
    """
    for part in fp.parts:
        if part.startswith("tb_"):
            if known_backends is None or part in known_backends:
                return part
    return None


def collect_records(results_root: Path, variants: List[str], backends: Optional[List[str]] = None) -> Tuple[List[Record], Dict[str, int]]:
    records: List[Record] = []
    stats = {
        "txt_files_found": 0,
        "matched_filenames": 0,
        "files_with_parsed_rows": 0,
        "total_rows_parsed": 0,
    }

    for fp in results_root.rglob("*.txt"):
        stats["txt_files_found"] += 1

        backend = infer_backend_from_path(fp, backends)
        if backend is None:
            continue

        parsed = parse_filename(fp)
        if not parsed:
            continue
        variant, bytes_, iters, _rep = parsed
        if variant not in variants:
            continue

        stats["matched_filenames"] += 1

        text = fp.read_text(encoding="utf-8", errors="replace")
        rows = parse_strace_c(text)
        if not rows:
            continue

        stats["files_with_parsed_rows"] += 1
        stats["total_rows_parsed"] += len(rows)

        for r in rows:
            records.append(
                Record(
                    backend=backend,
                    variant=variant,
                    bytes=bytes_,
                    iterations=iters,
                    syscall=str(r["syscall"]),
                    pct_time=float(r["pct_time"]),
                    calls=int(r["calls"]),
                )
            )

    return records, stats


def aggregate_repeats_mean(records: List[Record]) -> List[Record]:
    """
    Your input has multiple rep files for same (backend,variant,bytes,iters).
    strace -c already aggregates within a run, so here we aggregate across reps.
    We take mean(pct_time) and mean(calls) per syscall.
    Missing syscalls in a rep implicitly contribute 0 to the mean (handled by treating
    absent as 0 during aggregation).
    """
    # First collect the full syscall universe per (backend,variant,bytes,iters)
    key_run = lambda r: (r.backend, r.variant, r.bytes, r.iterations)
    runs: Dict[Tuple[str,str,int,int], set] = {}
    for r in records:
        k = key_run(r)
        runs.setdefault(k, set()).add(r.syscall)

    # Accumulate sums and counts-of-reps for each (run, syscall)
    # We don't have explicit rep id in Record. But we can infer reps by file count is tricky.
    # Better: estimate number of reps per run by counting distinct files earlier.
    # Since we don't store rep, we approximate by counting unique (backend,variant,bytes,iters, "file instance")
    # Not available now. So: compute means by dividing by number of "observations" per run-syscall is wrong.
    #
    # Fix: We'll compute mean over "rep files" by counting how many files existed for each run.
    # That requires collecting rep counts from filenames. We'll do that by rescanning filenames.
    #
    # Instead of rescanning, easiest is: do NOT attempt to infer reps here;
    # treat each parsed file as one rep, and count reps by (backend,variant,bytes,iters, filepath parent run_id + repN).
    #
    # Simpler: store rep in Record? We didn't. We'll rebuild a rep-aware structure by changing collector:
    # But since you want drop-in: we do a second pass over filesystem in main() to get rep counts and inject them.
    raise RuntimeError("Internal: aggregate_repeats_mean expects rep-aware counts; use aggregate_records_from_files().")


@dataclass(frozen=True)
class Rec2:
    backend: str
    variant: str
    bytes: int
    iterations: int
    syscall: str
    pct_time: float
    calls: float  # keep float for mean


def aggregate_records_from_files(results_root: Path, variants: List[str], backends: Optional[List[str]] = None) -> Tuple[List[Record], Dict[str,int], List[Rec2]]:
    """
    Collect records from files and also aggregate repeats properly:
    - identify each rep file as one rep
    - for each (backend,variant,bytes,iters), compute rep_count
    - for each syscall, sum over reps (missing syscall in a rep counts as 0), then divide by rep_count
    Returns:
      raw_records (per file rows), stats, aggregated_records (mean across reps)
    """
    raw_records: List[Record] = []
    stats = {
        "txt_files_found": 0,
        "matched_filenames": 0,
        "files_with_parsed_rows": 0,
        "total_rows_parsed": 0,
    }

    # rep counts per run (backend,variant,bytes,iters)
    rep_count: Dict[Tuple[str,str,int,int], int] = {}

    # temp sums per run+syscall
    sum_pct: Dict[Tuple[str,str,int,int,str], float] = {}
    sum_calls: Dict[Tuple[str,str,int,int,str], float] = {}

    # syscall universe per run for padding missing syscalls
    sys_universe: Dict[Tuple[str,str,int,int], set] = {}

    for fp in results_root.rglob("*.txt"):
        stats["txt_files_found"] += 1

        backend = infer_backend_from_path(fp, backends)
        if backend is None:
            continue

        parsed = parse_filename(fp)
        if not parsed:
            continue
        variant, bytes_, iters, _rep = parsed
        if variant not in variants:
            continue

        stats["matched_filenames"] += 1

        text = fp.read_text(encoding="utf-8", errors="replace")
        rows = parse_strace_c(text)
        if not rows:
            continue

        stats["files_with_parsed_rows"] += 1
        stats["total_rows_parsed"] += len(rows)

        run_key = (backend, variant, bytes_, iters)
        rep_count[run_key] = rep_count.get(run_key, 0) + 1

        # mark syscalls seen in this run (across reps)
        uni = sys_universe.setdefault(run_key, set())
        for r in rows:
            uni.add(str(r["syscall"]))

        # Add this rep's contributions to sums
        for r in rows:
            syscall = str(r["syscall"])
            k = (backend, variant, bytes_, iters, syscall)
            sum_pct[k] = sum_pct.get(k, 0.0) + float(r["pct_time"])
            sum_calls[k] = sum_calls.get(k, 0.0) + float(r["calls"])

            raw_records.append(
                Record(
                    backend=backend,
                    variant=variant,
                    bytes=bytes_,
                    iterations=iters,
                    syscall=syscall,
                    pct_time=float(r["pct_time"]),
                    calls=int(r["calls"]),
                )
            )

    # Now build aggregated (mean across reps, missing treated as 0)
    agg: List[Rec2] = []
    for run_key, syscalls in sys_universe.items():
        reps = rep_count.get(run_key, 1)
        backend, variant, bytes_, iters = run_key
        for syscall in syscalls:
            k = (backend, variant, bytes_, iters, syscall)
            pct = sum_pct.get(k, 0.0) / reps
            calls = sum_calls.get(k, 0.0) / reps
            agg.append(Rec2(backend, variant, bytes_, iters, syscall, pct, calls))

    return raw_records, stats, agg


def format_bytes(n: int) -> str:
    if n % MB == 0:
        return f"{n // MB}MB"
    if n % KB == 0:
        return f"{n // KB}KB"
    return f"{n}B"


def run_key(run: Tuple[int, int]) -> str:
    b, it = run
    return f"{format_bytes(b)}\n{it} iters"


def top_syscalls_by_mean_metric(
    records: List[Rec2],
    backend: str,
    variant: str,
    metric: str,
    top_k: int,
) -> List[str]:
    assert metric in ("pct_time", "calls")
    accum: Dict[str, Tuple[float, int]] = {}
    for r in records:
        if r.backend != backend or r.variant != variant:
            continue
        val = float(r.pct_time) if metric == "pct_time" else float(r.calls)
        tot, n = accum.get(r.syscall, (0.0, 0))
        accum[r.syscall] = (tot + val, n + 1)

    ranked = sorted(accum.items(), key=lambda kv: (kv[1][0] / max(kv[1][1], 1)), reverse=True)
    return [s for s, _ in ranked[:top_k]]


def build_run_table(
    records: List[Rec2],
    backend: str,
    variant: str,
    metric: str,
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], Dict[str, float]]]:
    assert metric in ("pct_time", "calls")
    runs = sorted(set((r.bytes, r.iterations) for r in records if r.backend == backend and r.variant == variant))
    if not runs:
        return [], {}

    run_vals: Dict[Tuple[int, int], Dict[str, float]] = {run: {} for run in runs}
    for r in records:
        if r.backend != backend or r.variant != variant:
            continue
        run = (r.bytes, r.iterations)
        val = float(r.pct_time) if metric == "pct_time" else float(r.calls)
        run_vals[run][r.syscall] = run_vals[run].get(r.syscall, 0.0) + val

    return runs, run_vals


def plot_overall_mix_per_run(
    records: List[Rec2],
    out_dir: Path,
    backend: str,
    variant: str,
    top_k: int,
    metric: str,
    normalize_calls_per_iter: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    runs, run_vals = build_run_table(records, backend, variant, metric)
    if not runs:
        return

    top_sys = top_syscalls_by_mean_metric(records, backend, variant, metric, top_k)

    xlabels = [run_key(r) for r in runs]
    x = list(range(len(runs)))

    def get_val(run: Tuple[int, int], syscall: str) -> float:
        _b, it = run
        v = run_vals[run].get(syscall, 0.0)
        if metric == "calls" and normalize_calls_per_iter:
            v = v / max(it, 1)
        return float(v)

    layers: List[Tuple[str, List[float]]] = []
    for s in top_sys:
        layers.append((s, [get_val(run, s) for run in runs]))

    other = []
    for run in runs:
        total = sum(get_val(run, s) for s in run_vals[run].keys())
        top_sum = sum(get_val(run, s) for s in top_sys)
        other.append(max(0.0, total - top_sum))
    layers.append(("other", other))

    plt.figure(figsize=(max(12, len(runs) * 0.8), 6))
    bottom = [0.0] * len(runs)
    for name, vals in layers:
        plt.bar(x, vals, bottom=bottom, label=name)
        bottom = [b + v for b, v in zip(bottom, vals)]

    plt.xticks(x, xlabels)

    if metric == "pct_time":
        plt.ylabel("% time")
        plt.title(f"{backend}/{variant}: overall syscall mix per run (% time)")
        out = out_dir / f"{variant}_mix_per_run_pct_time.png"
    else:
        plt.ylabel("calls/iter" if normalize_calls_per_iter else "calls")
        plt.title(f"{backend}/{variant}: overall syscall mix per run ({'calls/iter' if normalize_calls_per_iter else 'calls'})")
        out = out_dir / (f"{variant}_mix_per_run_calls_per_iter.png" if normalize_calls_per_iter else f"{variant}_mix_per_run_calls.png")

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")


def plot_heatmap_per_variant(
    records: List[Rec2],
    out_dir: Path,
    backend: str,
    variant: str,
    metric: str,
    top_k: int = 12,
    normalize_calls_per_iter: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    runs, run_vals = build_run_table(records, backend, variant, metric)
    if not runs:
        return

    top_sys = top_syscalls_by_mean_metric(records, backend, variant, metric, top_k)

    mat: List[List[float]] = []
    for s in top_sys:
        row = []
        for (b, it) in runs:
            v = run_vals[(b, it)].get(s, 0.0)
            if metric == "calls" and normalize_calls_per_iter:
                v = v / max(it, 1)
            row.append(float(v))
        mat.append(row)

    plt.figure(figsize=(max(10, len(runs) * 0.9), max(5, len(top_sys) * 0.5)))
    im = plt.imshow(mat, aspect="auto")

    plt.yticks(range(len(top_sys)), top_sys)
    plt.xticks(range(len(runs)), [run_key(r) for r in runs], rotation=0)

    if metric == "pct_time":
        plt.title(f"{backend}/{variant}: %time heatmap (top {top_k} syscalls)")
        cbar_label = "% time"
        fname = f"{variant}_heatmap_pct_time.png"
    else:
        label = "calls/iter" if normalize_calls_per_iter else "calls"
        plt.title(f"{backend}/{variant}: {label} heatmap (top {top_k} syscalls)")
        cbar_label = label
        fname = f"{variant}_heatmap_calls_per_iter.png" if normalize_calls_per_iter else f"{variant}_heatmap_calls.png"

    plt.colorbar(im, label=cbar_label)
    plt.tight_layout()
    out = out_dir / fname
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")


def plot_delta_to_baseline(
    records: List[Rec2],
    out_dir: Path,
    backend: str,
    variant: str,
    metric: str,
    top_k: int = 12,
    normalize_calls_per_iter: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    if variant == "baseline":
        return

    runs_v, vals_v = build_run_table(records, backend, variant, metric)
    runs_b, vals_b = build_run_table(records, backend, "baseline", metric)

    common_runs = sorted(set(runs_v).intersection(set(runs_b)))
    if not common_runs:
        return

    top_sys = top_syscalls_by_mean_metric(records, backend, variant, metric, top_k)

    out_dir.mkdir(parents=True, exist_ok=True)

    for (b, it) in common_runs:
        labels: List[str] = []
        deltas: List[float] = []

        for s in top_sys:
            vv = vals_v[(b, it)].get(s, 0.0)
            bb = vals_b[(b, it)].get(s, 0.0)
            if metric == "calls" and normalize_calls_per_iter:
                vv = vv / max(it, 1)
                bb = bb / max(it, 1)
            labels.append(s)
            deltas.append(float(vv - bb))

        order = sorted(range(len(deltas)), key=lambda i: abs(deltas[i]), reverse=True)
        labels = [labels[i] for i in order]
        deltas = [deltas[i] for i in order]

        plt.figure(figsize=(10, max(5, len(labels) * 0.35)))
        plt.barh(labels[::-1], deltas[::-1])
        plt.axvline(0.0)
        plt.tight_layout()

        if metric == "pct_time":
            plt.xlabel("delta %time (variant - baseline)")
            plt.title(f"{backend}: {variant} vs baseline %time — {format_bytes(b)}, {it} iters")
            fname = f"{variant}_delta_pct_{format_bytes(b)}_{it}iters.png"
        else:
            xlabel = "delta calls/iter" if normalize_calls_per_iter else "delta calls"
            plt.xlabel(f"{xlabel} (variant - baseline)")
            plt.title(f"{backend}: {variant} vs baseline {xlabel} — {format_bytes(b)}, {it} iters")
            fname = f"{variant}_delta_calls_{format_bytes(b)}_{it}iters.png"

        out = out_dir / fname
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True, help="Typically your repo root ./results")
    ap.add_argument("--variants", nargs="+", default=["alloc", "baseline", "kernels"])
    ap.add_argument("--backends", nargs="*", default=None, help="Optional list e.g. tb_cpu tb_cuda")
    ap.add_argument("--top", type=int, default=8, help="Top K syscalls for stacked/heatmap/delta")
    ap.add_argument("--out", type=Path, default=Path("./_strace_plots"))
    args = ap.parse_args()

    raw_records, stats, agg_records = aggregate_records_from_files(args.results_root, args.variants, args.backends)

    print(
        "Scan stats:",
        f"txt_files_found={stats['txt_files_found']},",
        f"matched_filenames={stats['matched_filenames']},",
        f"files_with_parsed_rows={stats['files_with_parsed_rows']},",
        f"total_rows_parsed={stats['total_rows_parsed']}",
    )

    if not agg_records:
        raise SystemExit("No records parsed.")

    args.out.mkdir(parents=True, exist_ok=True)

    # determine backends present
    present_backends = sorted(set(r.backend for r in agg_records))
    for backend in present_backends:
        for v in args.variants:
            vdir = args.out / backend / v

            plot_overall_mix_per_run(agg_records, vdir, backend, v, top_k=args.top, metric="pct_time")
            plot_overall_mix_per_run(agg_records, vdir, backend, v, top_k=args.top, metric="calls", normalize_calls_per_iter=True)

            plot_heatmap_per_variant(agg_records, vdir, backend, v, metric="pct_time", top_k=max(args.top, 12))
            plot_heatmap_per_variant(agg_records, vdir, backend, v, metric="calls", top_k=max(args.top, 12), normalize_calls_per_iter=True)

            plot_delta_to_baseline(agg_records, vdir, backend, v, metric="pct_time", top_k=max(args.top, 12))
            plot_delta_to_baseline(agg_records, vdir, backend, v, metric="calls", top_k=max(args.top, 12), normalize_calls_per_iter=True)

    print(f"Done. Plots under: {args.out.resolve()}")


if __name__ == "__main__":
    main()
