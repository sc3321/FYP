#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MB = 1024 * 1024
KB = 1024

# --- filenames like:
# alloc_10mb_10000iters_summary.txt
# 10mb_100iters_summary.txt
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


# --- tolerant strace -c row regex:
# Matches BOTH:
#  99.76 2.536047 253 10004 889 futex
#  0.05  0.001322 62  21          mmap   (errors blank in table -> effectively absent)
#  0.05  0.001322 62  21 mmap       (no errors column format)
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


def top_syscalls_by_mean_pct(records: List[Record], variant: str, top_k: int) -> List[str]:
    accum: Dict[str, Tuple[float, int]] = {}
    for r in records:
        if r.variant != variant:
            continue
        tot, n = accum.get(r.syscall, (0.0, 0))
        accum[r.syscall] = (tot + r.pct_time, n + 1)
    ranked = sorted(accum.items(), key=lambda kv: (kv[1][0] / max(kv[1][1], 1)), reverse=True)
    return [s for s, _ in ranked[:top_k]]


def plot_per_syscall_lines(records: List[Record], out_dir: Path, variant: str, syscalls: List[str]) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    by_key: Dict[Tuple[int, int, str], Tuple[float, int, int]] = {}
    for r in records:
        if r.variant != variant or r.syscall not in syscalls:
            continue
        key = (r.bytes, r.iterations, r.syscall)
        pct_sum, calls_sum, n = by_key.get(key, (0.0, 0, 0))
        by_key[key] = (pct_sum + r.pct_time, calls_sum + r.calls, n + 1)

    for syscall in syscalls:
        pts = [(b, it) for (b, it, s) in by_key.keys() if s == syscall]
        if not pts:
            continue

        iters_sorted = sorted(set(it for (b, it) in pts))
        bytes_sorted = sorted(set(b for (b, it) in pts))

        # %time vs bytes
        plt.figure()
        for it in iters_sorted:
            xs, ys = [], []
            for b in bytes_sorted:
                key = (b, it, syscall)
                if key not in by_key:
                    continue
                pct_sum, _, n = by_key[key]
                xs.append(b)
                ys.append(pct_sum / n)
            if xs:
                plt.plot(xs, ys, marker="o", label=f"{it} iters")
        plt.xscale("log")
        plt.xlabel("bytes allocated (log scale)")
        plt.ylabel("% time")
        plt.title(f"{variant}: {syscall} — %time vs bytes")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{variant}_{syscall}_pct_vs_bytes.png", dpi=200)
        plt.close()

        # calls vs bytes
        plt.figure()
        for it in iters_sorted:
            xs, ys = [], []
            for b in bytes_sorted:
                key = (b, it, syscall)
                if key not in by_key:
                    continue
                _, calls_sum, n = by_key[key]
                xs.append(b)
                ys.append(calls_sum // n)
            if xs:
                plt.plot(xs, ys, marker="o", label=f"{it} iters")
        plt.xscale("log")
        plt.xlabel("bytes allocated (log scale)")
        plt.ylabel("calls")
        plt.title(f"{variant}: {syscall} — calls vs bytes")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{variant}_{syscall}_calls_vs_bytes.png", dpi=200)
        plt.close()


def plot_overall_mix_per_run(records: List[Record], out_dir: Path, variant: str, top_k: int, metric: str) -> None:
    import matplotlib.pyplot as plt

    assert metric in ("pct_time", "calls")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = sorted(set((r.bytes, r.iterations) for r in records if r.variant == variant))
    if not runs:
        return

    # top_k syscalls by mean metric
    accum: Dict[str, Tuple[float, int]] = {}
    for r in records:
        if r.variant != variant:
            continue
        val = float(r.pct_time) if metric == "pct_time" else float(r.calls)
        tot, n = accum.get(r.syscall, (0.0, 0))
        accum[r.syscall] = (tot + val, n + 1)
    ranked = sorted(accum.items(), key=lambda kv: (kv[1][0] / max(kv[1][1], 1)), reverse=True)
    top_sys = [s for s, _ in ranked[:top_k]]

    run_vals: Dict[Tuple[int, int], Dict[str, float]] = {run: {} for run in runs}
    for r in records:
        if r.variant != variant:
            continue
        run = (r.bytes, r.iterations)
        val = float(r.pct_time) if metric == "pct_time" else float(r.calls)
        run_vals[run][r.syscall] = run_vals[run].get(r.syscall, 0.0) + val

    xlabels = [f"{format_bytes(b)}\n{it} iters" for (b, it) in runs]
    x = list(range(len(runs)))

    layers: List[Tuple[str, List[float]]] = []
    for s in top_sys:
        layers.append((s, [run_vals[run].get(s, 0.0) for run in runs]))

    other = []
    for run in runs:
        total = sum(run_vals[run].values())
        top_sum = sum(run_vals[run].get(s, 0.0) for s in top_sys)
        other.append(max(0.0, total - top_sum))
    layers.append(("other", other))

    plt.figure(figsize=(max(12, len(runs) * 0.7), 6))
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
        plt.ylabel("calls")
        plt.title(f"{variant}: overall syscall mix per run (calls)")
        out = out_dir / f"{variant}_mix_per_run_calls.png"

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=["alloc", "baseline", "kernels"])
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("./_strace_plots"))
    args = ap.parse_args()

    records, stats = collect_records(args.results_root, args.variants)
    print(
        "Scan stats:",
        f"txt_files_found={stats['txt_files_found']},",
        f"matched_param_filenames={stats['matched_param_filenames']},",
        f"files_with_parsed_rows={stats['files_with_parsed_rows']},",
        f"total_rows_parsed={stats['total_rows_parsed']}",
    )

    if not records:
        raise SystemExit("No records parsed. The scan stats above should tell us which stage failed.")

    args.out.mkdir(parents=True, exist_ok=True)

    for v in args.variants:
        top_sys = top_syscalls_by_mean_pct(records, v, args.top)
        if not top_sys:
            print(f"Skipping {v}: no data")
            continue

        plot_per_syscall_lines(records, args.out / v, v, top_sys)
        plot_overall_mix_per_run(records, args.out / v, v, top_k=args.top, metric="pct_time")
        plot_overall_mix_per_run(records, args.out / v, v, top_k=args.top, metric="calls")

    print(f"Done. Plots under: {args.out.resolve()}")


if __name__ == "__main__":
    main()

