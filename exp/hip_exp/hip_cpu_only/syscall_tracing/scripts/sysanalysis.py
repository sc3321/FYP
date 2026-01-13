#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ----------------------------
# Parsing utilities (from your current code)
# ----------------------------
import re

MB = 1024 * 1024
KB = 1024

PARAM_RE = re.compile(
    r"(?P<size>\d+)\s*(?P<unit>mb|kb|b)(?:[^0-9A-Za-z]|_).*?(?P<iters>\d+)\s*(?:iters|iter)",
    re.IGNORECASE,
)

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

def parse_strace_c(text: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("% time"):
            continue
        if set(line) <= {"-", " "}:
            continue
        # skip the "total" row (different format)
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
    seconds: float
    usecs_per_call: int

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
                    seconds=float(r["seconds"]),
                    usecs_per_call=int(r["usecs_per_call"]),
                )
            )

    return records, stats


# ----------------------------
# Analysis layer
# ----------------------------

def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")

def log_ratio(a: float, b: float) -> float:
    # returns ln(a/b) if both positive else nan
    if a > 0 and b > 0:
        return math.log(a) - math.log(b)
    return float("nan")

def elasticity(y1: float, y2: float, x1: float, x2: float) -> float:
    # E = Δlog(y)/Δlog(x)
    num = log_ratio(y2, y1)
    den = log_ratio(x2, x1)
    if math.isnan(num) or math.isnan(den) or den == 0:
        return float("nan")
    return num / den

def bucket_syscall(syscall: str, custom: Optional[Dict[str, str]] = None) -> str:
    if custom and syscall in custom:
        return custom[syscall]
    s = syscall.lower()
    # default coarse buckets
    if s in {"mmap", "munmap", "brk", "mprotect", "madvise"}:
        return "memory"
    if s in {"futex", "sched_yield", "nanosleep", "clock_nanosleep"}:
        return "sync"
    if s in {"ioctl", "poll", "ppoll", "epoll_wait", "epoll_pwait", "select", "pselect6"}:
        return "control_io"
    if s in {"read", "write", "pread64", "pwrite64", "openat", "close", "lseek", "fcntl"}:
        return "file_io"
    return "other"

@dataclass
class AggRow:
    variant: str
    bytes: int
    iterations: int
    key: str  # syscall or category
    calls: int
    seconds: float
    pct_time: float  # sum of pct_time across included syscalls (composition metric only)

    # derived
    calls_per_iter: float
    usec_per_iter: float
    usec_per_call: float
    
    calls_per_byte: float
    usec_per_byte: float

def aggregate(records: List[Record], by: str, bucket_map: Optional[Dict[str, str]]) -> List[AggRow]:
    """
    by: "syscall" or "category"
    """
    acc: Dict[Tuple[str, int, int, str], Dict[str, float]] = defaultdict(lambda: {"calls": 0.0, "seconds": 0.0, "pct_time": 0.0})
    for r in records:
        key = r.syscall if by == "syscall" else bucket_syscall(r.syscall, bucket_map)
        k = (r.variant, r.bytes, r.iterations, key)
        acc[k]["calls"] += r.calls
        acc[k]["seconds"] += r.seconds
        acc[k]["pct_time"] += r.pct_time

    out: List[AggRow] = []
    for (variant, bytes_, iters, key), v in sorted(acc.items()):
        calls = int(round(v["calls"]))
        seconds = float(v["seconds"])
        pct_time = float(v["pct_time"])

        calls_per_iter = safe_div(calls, iters)
        usec_per_iter = safe_div(seconds * 1e6, iters)
        usec_per_call = safe_div(seconds * 1e6, calls)

        calls_per_byte = safe_div(calls, bytes_)
        usec_per_byte = safe_div(seconds * 1e6, bytes_)


        out.append(
            AggRow(
                variant=variant,
                bytes=bytes_,
                iterations=iters,
                key=key,
                calls=calls,
                seconds=seconds,
                pct_time=pct_time,
                calls_per_iter=calls_per_iter,
                usec_per_iter=usec_per_iter,
                usec_per_call=usec_per_call,
                calls_per_byte=calls_per_byte,
                usec_per_byte=usec_per_byte,
            )
        )
    return out

def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def top_n_by_metric(agg: List[AggRow], metric: str, n: int) -> List[AggRow]:
    return sorted(agg, key=lambda r: (getattr(r, metric) if not math.isnan(getattr(r, metric)) else -1e30), reverse=True)[:n]

def compute_elasticities(
    agg: List[AggRow],
    focus: str,
    focus_values: List[int],
    vary: str,
    metric: str,
) -> List[Dict[str, object]]:
    """
    elasticity of metric vs vary, for each (variant, key, focus_value) over adjacent points in vary.
    focus: "bytes" or "iterations" (held fixed)
    vary:  "iterations" or "bytes" (varied)
    focus_values: list of exact held values to evaluate (e.g. bytes=[1e6,5e6,1e7])
    """
    # index: (variant,key,focus_value) -> list of (vary_value, metric_value)
    idx: Dict[Tuple[str, str, int], List[Tuple[int, float]]] = defaultdict(list)
    for r in agg:
        fv = r.bytes if focus == "bytes" else r.iterations
        vv = r.iterations if vary == "iterations" else r.bytes
        if fv in focus_values:
            idx[(r.variant, r.key, fv)].append((vv, getattr(r, metric)))

    out: List[Dict[str, object]] = []
    for (variant, key, fv), pts in idx.items():
        pts_sorted = sorted(pts, key=lambda t: t[0])
        for (x1, y1), (x2, y2) in zip(pts_sorted, pts_sorted[1:]):
            e = elasticity(y1, y2, x1, x2)
            out.append(
                {
                    "variant": variant,
                    "key": key,
                    focus: fv,
                    "vary_x1": x1,
                    "vary_x2": x2,
                    "metric": metric,
                    "y1": y1,
                    "y2": y2,
                    "elasticity": e,
                }
            )
    return out

def classify_key(elasticity_iters: float, elasticity_bytes: float) -> str:
    # crude, interpretable bins
    # (tune thresholds later if you want)
    def ok(x: float) -> bool:
        return not math.isnan(x) and math.isfinite(x)

    ei = elasticity_iters if ok(elasticity_iters) else None
    eb = elasticity_bytes if ok(elasticity_bytes) else None

    if ei is None and eb is None:
        return "unknown"
    if ei is not None and (eb is None or abs(eb) < 0.25) and ei > 0.75:
        return "iteration_scaled"
    if eb is not None and (ei is None or abs(ei) < 0.25) and eb > 0.75:
        return "byte_scaled"
    if ei is not None and eb is not None and ei > 0.5 and eb > 0.5:
        return "both_scaled"
    if (ei is not None and abs(ei) < 0.25) and (eb is not None and abs(eb) < 0.25):
        return "mostly_constant"
    return "mixed"

def summarise_classification(
    elast_iters: List[Dict[str, object]],
    elast_bytes: List[Dict[str, object]],
    focus_bytes: List[int],
    focus_iters: List[int],
) -> List[Dict[str, object]]:
    """
    Produce one classification per (variant,key) by averaging available elasticities.
    """
    # Average elasticities by (variant,key)
    def mean(xs: List[float]) -> float:
        xs2 = [x for x in xs if not math.isnan(x) and math.isfinite(x)]
        return sum(xs2) / len(xs2) if xs2 else float("nan")

    e_i: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    e_b: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for r in elast_iters:
        e_i[(r["variant"], r["key"])].append(float(r["elasticity"]))
    for r in elast_bytes:
        e_b[(r["variant"], r["key"])].append(float(r["elasticity"]))

    out: List[Dict[str, object]] = []
    keys = set(e_i.keys()) | set(e_b.keys())
    for vk in sorted(keys):
        variant, key = vk
        mi = mean(e_i.get(vk, []))
        mb = mean(e_b.get(vk, []))
        out.append(
            {
                "variant": variant,
                "key": key,
                "E_iters_mean": mi,
                "E_bytes_mean": mb,
                "class": classify_key(mi, mb),
            }
        )
    return out


# ----------------------------
# CLI
# ----------------------------

def load_bucket_map(path: Optional[Path]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    m: Dict[str, str] = {}
    # CSV format: syscall,category
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sc = (row.get("syscall") or "").strip()
            cat = (row.get("category") or "").strip()
            if sc and cat:
                m[sc] = cat
    return m

def parse_int_list(s: str) -> List[int]:
    # accepts "1000000,5000000,10000000"
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description="Numerical analysis for strace -c sweeps (elasticities, per-iter metrics).")
    ap.add_argument("--results-root", type=Path, required=True, help="Root directory containing strace .txt outputs.")
    ap.add_argument("--variants", type=str, default="baseline,alloc,kernels", help="Comma-separated variant folder names.")
    ap.add_argument("--bytes", type=str, default="1000000,5000000,10000000", help="Comma-separated bytes sweep points.")
    ap.add_argument("--iters", type=str, default="100,5000,100000", help="Comma-separated iteration sweep points.")
    ap.add_argument("--bucket-map", type=Path, default=None, help="Optional CSV mapping: syscall,category")
    ap.add_argument("--outdir", type=Path, default=Path("analysis_out"), help="Output directory.")
    ap.add_argument("--topn", type=int, default=15, help="Top N syscalls/categories to dump by time_per_iter.")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bytes_points = parse_int_list(args.bytes)
    iters_points = parse_int_list(args.iters)

    bucket_map = load_bucket_map(args.bucket_map)

    records, stats = collect_records(args.results_root, variants)

    print("== Parse stats ==")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"records: {len(records)}")
    if not records:
        print("No records parsed. Check --results-root, --variants, and filename pattern.")
        return 2

    # Aggregate
    agg_sys = aggregate(records, by="syscall", bucket_map=bucket_map)
    agg_cat = aggregate(records, by="category", bucket_map=bucket_map)

    # Write aggregate tables
    def agg_to_dict(r: AggRow) -> Dict[str, object]:
        return {
            "variant": r.variant,
            "bytes": r.bytes,
            "iterations": r.iterations,
            "key": r.key,
            "calls": r.calls,
            "seconds": r.seconds,
            "pct_time_sum": r.pct_time,
            "calls_per_iter": r.calls_per_iter,
            "usec_per_iter": r.usec_per_iter,
            "usec_per_call": r.usec_per_call,
            "calls_per_byte": r.calls_per_byte,
            "usec_per_byte": r.usec_per_byte,
        }

    write_csv(
        args.outdir / "aggregate_syscall.csv",
        (agg_to_dict(r) for r in agg_sys),
        ["variant", "bytes", "iterations", "key", "calls", "seconds", "pct_time_sum", "calls_per_iter", "usec_per_iter", "usec_per_call", "calls_per_byte", "usec_per_byte"],
    )
    write_csv(
        args.outdir / "aggregate_category.csv",
        (agg_to_dict(r) for r in agg_cat),
        ["variant", "bytes", "iterations", "key", "calls", "seconds", "pct_time_sum", "calls_per_iter", "usec_per_iter", "usec_per_call", "calls_per_byte", "usec_per_byte"],
    )

    # Elasticities (usec_per_iter is your main “cost” metric)
    elast_iters_sys = compute_elasticities(
        agg_sys, focus="bytes", focus_values=bytes_points, vary="iterations", metric="usec_per_iter"
    )
    elast_bytes_sys = compute_elasticities(
        agg_sys, focus="iterations", focus_values=iters_points, vary="bytes", metric="usec_per_iter"
    )

    elast_iters_cat = compute_elasticities(
        agg_cat, focus="bytes", focus_values=bytes_points, vary="iterations", metric="usec_per_iter"
    )
    elast_bytes_cat = compute_elasticities(
        agg_cat, focus="iterations", focus_values=iters_points, vary="bytes", metric="usec_per_iter"
    )

    write_csv(
        args.outdir / "elasticity_iters_syscall.csv",
        elast_iters_sys,
        ["variant", "key", "bytes", "vary_x1", "vary_x2", "metric", "y1", "y2", "elasticity"],
    )
    write_csv(
        args.outdir / "elasticity_bytes_syscall.csv",
        elast_bytes_sys,
        ["variant", "key", "iterations", "vary_x1", "vary_x2", "metric", "y1", "y2", "elasticity"],
    )
    write_csv(
        args.outdir / "elasticity_iters_category.csv",
        elast_iters_cat,
        ["variant", "key", "bytes", "vary_x1", "vary_x2", "metric", "y1", "y2", "elasticity"],
    )
    write_csv(
        args.outdir / "elasticity_bytes_category.csv",
        elast_bytes_cat,
        ["variant", "key", "iterations", "vary_x1", "vary_x2", "metric", "y1", "y2", "elasticity"],
    )

    # Classification (averaged elasticities)
    class_sys = summarise_classification(elast_iters_sys, elast_bytes_sys, bytes_points, iters_points)
    class_cat = summarise_classification(elast_iters_cat, elast_bytes_cat, bytes_points, iters_points)

    write_csv(
        args.outdir / "classification_syscall.csv",
        class_sys,
        ["variant", "key", "E_iters_mean", "E_bytes_mean", "class"],
    )
    write_csv(
        args.outdir / "classification_category.csv",
        class_cat,
        ["variant", "key", "E_iters_mean", "E_bytes_mean", "class"],
    )

    # Dump a quick “top N” report per variant/config (syscalls + categories)
    def dump_top(agg: List[AggRow], label: str) -> None:
        print(f"\n== Top {args.topn} by usec_per_iter ({label}) ==")
        # group by (variant,bytes,iters)
        groups: Dict[Tuple[str, int, int], List[AggRow]] = defaultdict(list)
        for r in agg:
            if (r.bytes in bytes_points) and (r.iterations in iters_points):
                groups[(r.variant, r.bytes, r.iterations)].append(r)

        for (variant, b, it), rows in sorted(groups.items()):
            top = top_n_by_metric(rows, "usec_per_iter", args.topn)
            print(f"\n-- {variant} | bytes={b} | iters={it} --")
            for r in top:
                print(f"{r.key:20s}  usec/iter={r.usec_per_iter:10.2f}  calls/iter={r.calls_per_iter:10.4f}  usec/call={r.usec_per_call:10.2f}")

    dump_top(agg_sys, "syscall")
    dump_top(agg_cat, "category")

    print(f"\nWrote analysis CSVs to: {args.outdir.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

