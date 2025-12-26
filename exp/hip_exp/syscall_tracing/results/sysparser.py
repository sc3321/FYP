#!/usr/bin/env python3
"""
sysparser.py

Batch-parse `strace -c` summary files under a results directory and generate
comparative plots that are actually useful for small parameter sweeps.

For each variant (alloc/baseline/kernels) it generates:
  1) Stacked bar: %time mix per run (bytes, iters)
  2) Stacked bar: calls-per-iter mix per run (bytes, iters)
  3) Heatmap: %time of top syscalls (rows=syscalls, cols=runs)
  4) Heatmap: calls/iter of top syscalls
  5) Delta-to-baseline bars (alloc vs baseline, kernels vs baseline):
       - per run: delta %time per syscall
       - per run: delta calls/iter per syscall

Assumes filenames contain size+unit and iters, e.g.:
  alloc_10mb_10000iters_summary.txt
  10mb_100iters_summary.txt
  1048576b_100iters_summary.txt

Run from inside results/:
  python3 sysparser.py --results-root . --out ./_strace_plots --top 8
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MB = 1024 * 1024
KB = 1024

# FIXED: no \b after unit because '_' is a word char, so 'mb_' breaks word boundary
PARAM_RE = re.compile(
    r"(?P<size>\d+)\s*(?P<unit>mb|kb|b)(?:[^0-9A-Za-z]|_).*?(?P<iters>\d+)\s*(?:iters|iter)",
    re.IGNORECASE,
)

def parse_params_from_name(filename: str) -> Optional[Tuple[int, int]]:
    m = PARAM_RE.search(filename)
    if not m:
        return None
    size = int(m.group("size"))
    unit = m.group("unit").lower()
    iters = int(m.group("iters"))

    if unit == "mb":
        bytes_ = size * MB
    elif unit == "kb":
        bytes_ = size * KB
    else:
        bytes_ = size
    return bytes_, iters


# Tolerant strace -c row regex:
# Works with/without errors column, and with blank errors cells (which become "absent" on split).
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
    variant: str
    bytes: int
    iterations: int
    syscall: str
    pct_time: float
    calls: int


def infer_variant(path: Path, variants: List[str]) -> str:
    for part in path.parts:
        if part in variants:
            return part
    return path.parent.name


def collect_records(results_root: Path, variants: List[str]) -> Tuple[List[Record], Dict[str, int]]:
    records: List[Record] = []
    stats = {
        "txt_files_found": 0,
        "matched_param_filenames": 0,
        "files_with_parsed_rows": 0,
        "total_rows_parsed": 0,
    }

    for fp in results_root.rglob("*.txt"):
        stats["txt_files_found"] += 1

        variant = infer_variant(fp, variants)
        if variant not in variants:
            continue

        params = parse_params_from_name(fp.name)
        if not params:
            continue
        stats["matched_param_filenames"] += 1
        bytes_, iters = params

        text = fp.read_text(encoding="utf-8", errors="replace")
        rows = parse_strace_c(text)
        if not rows:
            continue

        stats["files_with_parsed_rows"] += 1
        stats["total_rows_parsed"] += len(rows)

        for r in rows:
            records.append(
                Record(
                    variant=variant,
                    bytes=bytes_,
                    iterations=iters,
                    syscall=str(r["syscall"]),
                    pct_time=float(r["pct_time"]),
                    calls=int(r["calls"]),
                )
            )

    return records, stats


def format_bytes(n: int) -> str:
    if n % MB == 0:
        return f"{n // MB}MB"
    if n % KB == 0:
        return f"{n // KB}KB"
    return f"{n}B"


def run_key(run: Tuple[int, int]) -> str:
    b, it = run
    return f"{format_bytes(b)}\n{it} iters"


def top_syscalls_by_mean_metric(records: List[Record], variant: str, metric: str, top_k: int) -> List[str]:
    assert metric in ("pct_time", "calls")
    accum: Dict[str, Tuple[float, int]] = {}
    for r in records:
        if r.variant != variant:
            continue
        val = float(r.pct_time) if metric == "pct_time" else float(r.calls)
        tot, n = accum.get(r.syscall, (0.0, 0))
        accum[r.syscall] = (tot + val, n + 1)

    ranked = sorted(accum.items(), key=lambda kv: (kv[1][0] / max(kv[1][1], 1)), reverse=True)
    return [s for s, _ in ranked[:top_k]]


def build_run_table(
    records: List[Record],
    variant: str,
    metric: str,  # "pct_time" or "calls"
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], Dict[str, float]]]:
    assert metric in ("pct_time", "calls")
    runs = sorted(set((r.bytes, r.iterations) for r in records if r.variant == variant))
    if not runs:
        return [], {}

    run_vals: Dict[Tuple[int, int], Dict[str, float]] = {run: {} for run in runs}
    for r in records:
        if r.variant != variant:
            continue
        run = (r.bytes, r.iterations)
        val = float(r.pct_time) if metric == "pct_time" else float(r.calls)
        run_vals[run][r.syscall] = run_vals[run].get(r.syscall, 0.0) + val

    return runs, run_vals


def plot_overall_mix_per_run(
    records: List[Record],
    out_dir: Path,
    variant: str,
    top_k: int,
    metric: str,  # "pct_time" or "calls"
    normalize_calls_per_iter: bool = True,
) -> None:
    """
    Stacked bars: per run (bytes,iters), show syscall mix.
    For calls, default is calls/iter so different iteration counts compare fairly.
    """
    import matplotlib.pyplot as plt

    assert metric in ("pct_time", "calls")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs, run_vals = build_run_table(records, variant, metric)
    if not runs:
        return

    top_sys = top_syscalls_by_mean_metric(records, variant, metric, top_k)

    xlabels = [run_key(r) for r in runs]
    x = list(range(len(runs)))

    def get_val(run: Tuple[int, int], syscall: str) -> float:
        b, it = run
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
        plt.title(f"{variant}: overall syscall mix per run (% time)")
        out = out_dir / f"{variant}_mix_per_run_pct_time.png"
    else:
        plt.ylabel("calls/iter" if normalize_calls_per_iter else "calls")
        plt.title(f"{variant}: overall syscall mix per run ({'calls/iter' if normalize_calls_per_iter else 'calls'})")
        out = out_dir / f"{variant}_mix_per_run_calls_per_iter.png" if normalize_calls_per_iter else out_dir / f"{variant}_mix_per_run_calls.png"

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")


def plot_heatmap_per_variant(
    records: List[Record],
    out_dir: Path,
    variant: str,
    metric: str,     # "pct_time" or "calls"
    top_k: int = 12,
    normalize_calls_per_iter: bool = True,
) -> None:
    """
    Heatmap: rows=syscalls (top_k), cols=runs (bytes,iters).
    """
    import matplotlib.pyplot as plt

    assert metric in ("pct_time", "calls")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs, run_vals = build_run_table(records, variant, metric)
    if not runs:
        return

    top_sys = top_syscalls_by_mean_metric(records, variant, metric, top_k)

    mat: List[List[float]] = []
    for s in to

