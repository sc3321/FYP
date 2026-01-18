#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# =============================================================================
# Parsing
# =============================================================================

# New filename format from your sweep script:
#   alloc_1000000b_5000iters_rep1.txt
#   kernels_10485760b_10000iters_rep2.txt
FNAME_RE = re.compile(
    r"^(?P<variant>alloc|baseline|kernels)_(?P<bytes>\d+)b_(?P<iters>\d+)iters_rep(?P<rep>\d+)\.txt$",
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


def parse_filename(fp: Path) -> Optional[Tuple[str, int, int, int]]:
    """
    Returns (variant, bytes, iters, rep) or None if not matching.
    """
    m = FNAME_RE.match(fp.name)
    if not m:
        return None
    return (
        m.group("variant").lower(),
        int(m.group("bytes")),
        int(m.group("iters")),
        int(m.group("rep")),
    )


def infer_backend(fp: Path) -> Optional[str]:
    """
    Expects results layout:
      results/tb_cpu/run_xxx/alloc_...txt
      results/tb_cuda/run_xxx/...
    Returns "tb_cpu" / "tb_cuda" / "tb_hip" if present in path.
    """
    for part in fp.parts:
        if part.startswith("tb_"):
            return part
    return None


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
    rep: int
    syscall: str
    pct_time: float
    calls: int
    seconds: float


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

        backend = infer_backend(fp)
        if backend is None:
            continue
        if backends is not None and backend not in backends:
            continue

        parsed = parse_filename(fp)
        if not parsed:
            continue
        variant, bytes_, iters, rep = parsed
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
                    rep=rep,
                    syscall=str(r["syscall"]),
                    pct_time=float(r["pct_time"]),
                    calls=int(r["calls"]),
                    seconds=float(r["seconds"]),
                )
            )

    return records, stats


# =============================================================================
# Aggregation + metrics
# =============================================================================

def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def log_ratio(a: float, b: float) -> float:
    if a > 0 and b > 0:
        return math.log(a) - math.log(b)
    return float("nan")


def elasticity(y1: float, y2: float, x1: float, x2: float) -> float:
    num = log_ratio(y2, y1)
    den = log_ratio(x2, x1)
    if math.isnan(num) or math.isnan(den) or den == 0:
        return float("nan")
    return num / den


def bucket_syscall(syscall: str, custom: Optional[Dict[str, str]] = None) -> str:
    if custom and syscall in custom:
        return custom[syscall]
    s = syscall.lower()
    if s in {"mmap", "munmap", "brk", "mprotect", "madvise"}:
        return "memory"
    if s in {"futex", "sched_yield", "nanosleep", "clock_nanosleep"}:
        return "sync"
    if s in {"ioctl", "poll", "ppoll", "epoll_wait", "epoll_pwait", "select", "pselect6"}:
        return "control_io"
    if s in {"read", "write", "pread64", "pwrite64", "openat", "close", "lseek", "fcntl"}:
        return "file_io"
    return "other"


@dataclass(frozen=True)
class AggRow:
    backend: str
    variant: str
    bytes: int
    iterations: int
    key: str  # syscall or category
    calls: float
    seconds: float
    pct_time: float

    # derived
    calls_per_iter: float
    usec_per_iter: float
    usec_per_call: float
    calls_per_byte: float
    usec_per_byte: float


def mean(xs: List[float]) -> float:
    xs2 = [x for x in xs if not math.isnan(x) and math.isfinite(x)]
    return sum(xs2) / len(xs2) if xs2 else float("nan")


def aggregate_mean_over_reps(
    records: List[Record],
    by: str,  # "syscall" or "category"
    bucket_map: Optional[Dict[str, str]],
) -> List[AggRow]:
    """
    Two-stage aggregation:
      1) aggregate within each rep-file run to (backend,variant,bytes,iters,rep,key)
      2) mean over reps for each (backend,variant,bytes,iters,key)

    Important: if a syscall/category is absent in a rep, we treat it as 0 for that rep.
    """
    assert by in ("syscall", "category")

    # Universe of keys per (backend,variant,bytes,iters)
    universe: Dict[Tuple[str, str, int, int], set] = defaultdict(set)

    # Stage 1 accum per rep
    rep_acc: Dict[Tuple[str, str, int, int, int, str], Dict[str, float]] = defaultdict(
        lambda: {"calls": 0.0, "seconds": 0.0, "pct_time": 0.0}
    )

    # Count reps per (backend,variant,bytes,iters)
    rep_counts: Dict[Tuple[str, str, int, int], set] = defaultdict(set)

    for r in records:
        key = r.syscall if by == "syscall" else bucket_syscall(r.syscall, bucket_map)
        run_key = (r.backend, r.variant, r.bytes, r.iterations)
        universe[run_key].add(key)
        rep_counts[run_key].add(r.rep)

        k = (r.backend, r.variant, r.bytes, r.iterations, r.rep, key)
        rep_acc[k]["calls"] += float(r.calls)
        rep_acc[k]["seconds"] += float(r.seconds)
        rep_acc[k]["pct_time"] += float(r.pct_time)

    # Stage 2: mean over reps (including implicit zeros)
    out: List[AggRow] = []

    # Iterate runs deterministically
    for run_key in sorted(universe.keys()):
        backend, variant, bytes_, iters = run_key
        reps = sorted(rep_counts.get(run_key, {1}))
        nreps = len(reps) if reps else 1

        for key in sorted(universe[run_key]):
            calls_sum = 0.0
            sec_sum = 0.0
            pct_sum = 0.0

            for rep in reps:
                k = (backend, variant, bytes_, iters, rep, key)
                v = rep_acc.get(k)
                if v is None:
                    # absent in this rep -> 0
                    continue
                calls_sum += v["calls"]
                sec_sum += v["seconds"]
                pct_sum += v["pct_time"]

            calls = calls_sum / nreps
            seconds = sec_sum / nreps
            pct_time = pct_sum / nreps

            calls_per_iter = safe_div(calls, iters)
            usec_per_iter = safe_div(seconds * 1e6, iters)
            usec_per_call = safe_div(seconds * 1e6, calls)

            calls_per_byte = safe_div(calls, bytes_)
            usec_per_byte = safe_div(seconds * 1e6, bytes_)

            out.append(
                AggRow(
                    backend=backend,
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


# =============================================================================
# Elasticities + classification
# =============================================================================

def compute_elasticities(
    agg: List[AggRow],
    focus: str,
    focus_values: List[int],
    vary: str,
    metric: str,
) -> List[Dict[str, object]]:
    """
    elasticity of metric vs vary, for each (backend,variant,key,focus_value) over adjacent points in vary.
    """
    idx: Dict[Tuple[str, str, str, int], List[Tuple[int, float]]] = defaultdict(list)

    for r in agg:
        fv = r.bytes if focus == "bytes" else r.iterations
        vv = r.iterations if vary == "iterations" else r.bytes
        if fv in focus_values:
            idx[(r.backend, r.variant, r.key, fv)].append((vv, float(getattr(r, metric))))

    out: List[Dict[str, object]] = []
    for (backend, variant, key, fv), pts in idx.items():
        pts_sorted = sorted(pts, key=lambda t: t[0])
        for (x1, y1), (x2, y2) in zip(pts_sorted, pts_sorted[1:]):
            e = elasticity(y1, y2, x1, x2)
            out.append(
                {
                    "backend": backend,
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
) -> List[Dict[str, object]]:
    def mean_list(xs: List[float]) -> float:
        xs2 = [x for x in xs if not math.isnan(x) and math.isfinite(x)]
        return sum(xs2) / len(xs2) if xs2 else float("nan")

    e_i: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)  # (backend,variant,key) -> [E]
    e_b: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)

    for r in elast_iters:
        e_i[(r["backend"], r["variant"], r["key"])].append(float(r["elasticity"]))
    for r in elast_bytes:
        e_b[(r["backend"], r["variant"], r["key"])].append(float(r["elasticity"]))

    out: List[Dict[str, object]] = []
    keys = set(e_i.keys()) | set(e_b.keys())
    for bk in sorted(keys):
        backend, variant, key = bk
        mi = mean_list(e_i.get(bk, []))
        mb = mean_list(e_b.get(bk, []))
        out.append(
            {
                "backend": backend,
                "variant": variant,
                "key": key,
                "E_iters_mean": mi,
                "E_bytes_mean": mb,
                "class": classify_key(mi, mb),
            }
        )
    return out


# =============================================================================
# IO helpers
# =============================================================================

def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_bucket_map(path: Optional[Path]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    m: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sc = (row.get("syscall") or "").strip()
            cat = (row.get("category") or "").strip()
            if sc and cat:
                m[sc] = cat
    return m


def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def aggrow_to_dict(r: AggRow) -> Dict[str, object]:
    return {
        "backend": r.backend,
        "variant": r.variant,
        "bytes": r.bytes,
        "iterations": r.iterations,
        "key": r.key,
        "calls_mean": r.calls,
        "seconds_mean": r.seconds,
        "pct_time_mean": r.pct_time,
        "calls_per_iter": r.calls_per_iter,
        "usec_per_iter": r.usec_per_iter,
        "usec_per_call": r.usec_per_call,
        "calls_per_byte": r.calls_per_byte,
        "usec_per_byte": r.usec_per_byte,
    }


def top_n_by_metric(rows: List[AggRow], metric: str, n: int) -> List[AggRow]:
    def val(r: AggRow) -> float:
        x = float(getattr(r, metric))
        return x if (not math.isnan(x) and math.isfinite(x)) else -1e30
    return sorted(rows, key=val, reverse=True)[:n]


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Numerical analysis for strace -c sweeps (with backend + repeat aggregation).")
    ap.add_argument("--results-root", type=Path, required=True, help="Root directory containing results/tb_*/run_*/ files.")
    ap.add_argument("--variants", type=str, default="baseline,alloc,kernels", help="Comma-separated variants.")
    ap.add_argument("--backends", type=str, default="", help="Optional comma-separated backends (e.g. tb_cpu,tb_cuda). Empty=auto.")
    ap.add_argument("--bytes", type=str, default="1000000,5000000,10000000", help="Comma-separated bytes points (for elasticities).")
    ap.add_argument("--iters", type=str, default="100,5000,100000", help="Comma-separated iteration points (for elasticities).")
    ap.add_argument("--bucket-map", type=Path, default=None, help="Optional CSV mapping: syscall,category")
    ap.add_argument("--outdir", type=Path, default=Path("results/artifacts/statistics"), help="Output directory.")
    ap.add_argument("--topn", type=int, default=15, help="Top N syscalls/categories to print by usec_per_iter.")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bytes_points = parse_int_list(args.bytes)
    iters_points = parse_int_list(args.iters)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()] or None
    bucket_map = load_bucket_map(args.bucket_map)

    records, stats = collect_records(args.results_root, variants, backends)

    print("== Parse stats ==")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"records: {len(records)}")
    if not records:
        print("No records parsed. Check --results-root and filename pattern.")
        return 2

    # Aggregate (mean across reps)
    agg_sys = aggregate_mean_over_reps(records, by="syscall", bucket_map=bucket_map)
    agg_cat = aggregate_mean_over_reps(records, by="category", bucket_map=bucket_map)

    # Write aggregate tables
    write_csv(
        args.outdir / "aggregate_syscall.csv",
        (aggrow_to_dict(r) for r in agg_sys),
        ["backend","variant","bytes","iterations","key","calls_mean","seconds_mean","pct_time_mean",
         "calls_per_iter","usec_per_iter","usec_per_call","calls_per_byte","usec_per_byte"],
    )
    write_csv(
        args.outdir / "aggregate_category.csv",
        (aggrow_to_dict(r) for r in agg_cat),
        ["backend","variant","bytes","iterations","key","calls_mean","seconds_mean","pct_time_mean",
         "calls_per_iter","usec_per_iter","usec_per_call","calls_per_byte","usec_per_byte"],
    )

    # Elasticities (usec_per_iter is the primary cost metric)
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
        ["backend","variant","key","bytes","vary_x1","vary_x2","metric","y1","y2","elasticity"],
    )
    write_csv(
        args.outdir / "elasticity_bytes_syscall.csv",
        elast_bytes_sys,
        ["backend","variant","key","iterations","vary_x1","vary_x2","metric","y1","y2","elasticity"],
    )
    write_csv(
        args.outdir / "elasticity_iters_category.csv",
        elast_iters_cat,
        ["backend","variant","key","bytes","vary_x1","vary_x2","metric","y1","y2","elasticity"],
    )
    write_csv(
        args.outdir / "elasticity_bytes_category.csv",
        elast_bytes_cat,
        ["backend","variant","key","iterations","vary_x1","vary_x2","metric","y1","y2","elasticity"],
    )

    # Classification (averaged elasticities)
    class_sys = summarise_classification(elast_iters_sys, elast_bytes_sys)
    class_cat = summarise_classification(elast_iters_cat, elast_bytes_cat)

    write_csv(
        args.outdir / "classification_syscall.csv",
        class_sys,
        ["backend","variant","key","E_iters_mean","E_bytes_mean","class"],
    )
    write_csv(
        args.outdir / "classification_category.csv",
        class_cat,
        ["backend","variant","key","E_iters_mean","E_bytes_mean","class"],
    )

    # Top-N report (by usec_per_iter) at the focus points
    def dump_top(agg: List[AggRow], label: str) -> None:
        print(f"\n== Top {args.topn} by usec_per_iter ({label}) ==")
        groups: Dict[Tuple[str, str, int, int], List[AggRow]] = defaultdict(list)
        for r in agg:
            if (r.bytes in bytes_points) and (r.iterations in iters_points):
                groups[(r.backend, r.variant, r.bytes, r.iterations)].append(r)

        for (backend, variant, b, it), rows in sorted(groups.items()):
            top = top_n_by_metric(rows, "usec_per_iter", args.topn)
            print(f"\n-- {backend} | {variant} | bytes={b} | iters={it} --")
            for r in top:
                print(
                    f"{r.key:20s}  usec/iter={r.usec_per_iter:10.2f}  "
                    f"calls/iter={r.calls_per_iter:10.4f}  usec/call={r.usec_per_call:10.2f}"
                )

    dump_top(agg_sys, "syscall")
    dump_top(agg_cat, "category")

    print(f"\nWrote analysis CSVs to: {args.outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
