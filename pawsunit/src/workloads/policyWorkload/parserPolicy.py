#!/usr/bin/env python3
"""
Directory patterns parsed:
  run/events/*
  run/lc_events/*
  run/be_events/*
  run/be1_events/*, run/be2_events/*, ...
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, Iterable, List, Optional, Tuple

EVENT_RE = re.compile(
    r"Event type = (?P<event>BEGIN|END): "
    r"PhaseId:\[(?P<pid>\d+), (?P<phase_id>\d+)\],"
    r"Thread Id: (?P<tid>\d+), "
    r"parent_id: (?P<parent_id>-?\d+), "
    r"depth: (?P<depth>\d+),\s+"
    r"Timestamp: (?P<sec>\d+) s (?P<nsec>\d+) ns, "
    r"phase type: \((?P<phase_type>[^)]*)\), "
    r"workload class: (?P<class>\w+)"
    r"(?:, granularity: (?P<granularity>\w+))?")

TOP_LEVEL_BE_PHASES = {"BE_LONG_BATCH", "BE_CHUNKED_BATCH"}
LC_PHASES_OF_INTEREST = {
    "LC_REQUEST",
    "LC_PREFILL_SYNC",
    "LC_DECODE_STEP",
    "LC_PREFILL_SUBMISSION",
}


# ---------------------------------------------------------------------------
# Basic stats
# ---------------------------------------------------------------------------


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0

    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)

    if lo == hi:
        return xs[lo]

    frac = k - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def phase_stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
            "min_ms": 0.0,
            "stdev_ms": 0.0,
        }

    return {
        "count": len(vals),
        "mean_ms": mean(vals),
        "p50_ms": percentile(vals, 50),
        "p95_ms": percentile(vals, 95),
        "p99_ms": percentile(vals, 99),
        "max_ms": max(vals),
        "min_ms": min(vals),
        "stdev_ms": stdev(vals) if len(vals) > 1 else 0.0,
    }


def pearson_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0

    mx = mean(xs)
    my = mean(ys)

    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5

    if den_x == 0 or den_y == 0:
        return 0.0

    return num / (den_x * den_y)


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------


def is_event_log_path(path: Path) -> bool:
    """
    Accept event log files from:
      events/
      lc_events/
      be_events/
      be1_events/, be2_events/, ...

    This intentionally checks directory names, not file names.
    """
    parts = set(path.parts)

    if "events" in parts or "lc_events" in parts or "be_events" in parts:
        return True

    return any(re.fullmatch(r"be\d+_events", part) for part in path.parts)


def parse_event_file(path: Path) -> List[Dict]:
    events: List[Dict] = []

    with open(path, "r", errors="replace") as f:
        for line in f:
            m = EVENT_RE.search(line)
            if not m:
                continue

            d = m.groupdict()
            ts_ns = int(d["sec"]) * 1_000_000_000 + int(d["nsec"])

            events.append({
                "event": d["event"],
                "pid": int(d["pid"]),
                "tid": int(d["tid"]),
                "phase_id": int(d["phase_id"]),
                "parent_id": int(d["parent_id"]),
                "depth": int(d["depth"]),
                "ts_ns": ts_ns,
                "phase_type": d["phase_type"],
                "class": d["class"],
                "granularity": d.get("granularity") or "UNK",
                "file": str(path),
            })

    return events


def collect_events(run_dir: Path) -> List[Dict]:
    events: List[Dict] = []

    for path in run_dir.rglob("*"):
        if path.is_file() and is_event_log_path(path):
            events.extend(parse_event_file(path))

    return events


def pair_phases(events: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    begins: Dict[Tuple[int, int], Dict] = {}
    completed: List[Dict] = []
    unmatched_ends: List[Dict] = []
    duplicate_begins: List[Dict] = []

    for e in sorted(events, key=lambda x: x["ts_ns"]):
        key = (e["pid"], e["phase_id"])

        if e["event"] == "BEGIN":
            if key in begins:
                duplicate_begins.append(e)
            begins[key] = e
            continue

        if e["event"] == "END":
            b = begins.pop(key, None)
            if b is None:
                unmatched_ends.append(e)
                continue

            dur_ns = e["ts_ns"] - b["ts_ns"]
            completed.append({
                "pid": b["pid"],
                "tid": b["tid"],
                "phase_id": b["phase_id"],
                "parent_id": b["parent_id"],
                "depth": b["depth"],
                "phase_type": b["phase_type"],
                "class": b["class"],
                "begin_ns": b["ts_ns"],
                "end_ns": e["ts_ns"],
                "duration_ns": dur_ns,
                "duration_ms": dur_ns / 1_000_000.0,
                "file": b["file"],
            })

    unmatched_begins = list(begins.values())
    return completed, unmatched_begins, unmatched_ends, duplicate_begins


# ---------------------------------------------------------------------------
# Phase summaries
# ---------------------------------------------------------------------------


def summarise_by_phase(phases: List[Dict]) -> List[Dict]:
    by_key: Dict[Tuple[str, str], List[float]] = {}

    for p in phases:
        key = (p["class"], p["phase_type"])
        by_key.setdefault(key, []).append(p["duration_ms"])

    rows: List[Dict] = []
    for (klass, phase_type), vals in sorted(by_key.items()):
        rows.append({"class": klass, "phase_type": phase_type, **phase_stats(vals)})

    return rows


def lc_request_summary(run_name: str, phases: List[Dict]) -> Optional[Dict]:
    vals = [
        p["duration_ms"]
        for p in phases
        if p["class"] == "LC" and p["phase_type"] == "LC_REQUEST"
    ]

    if not vals:
        return None

    return {"run": run_name, **phase_stats(vals)}


def lc_phase_summary_rows(run_name: str, phases: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []

    for phase_type in sorted(LC_PHASES_OF_INTEREST):
        vals = [
            p["duration_ms"]
            for p in phases
            if p["class"] == "LC" and p["phase_type"] == phase_type
        ]
        if vals:
            rows.append({"run": run_name, "phase_type": phase_type, **phase_stats(vals)})

    return rows


def select_top_level_be_phases(phases: List[Dict]) -> List[Dict]:
    """
    Prefer top-level BE_LONG_BATCH / BE_CHUNKED_BATCH for throughput and overlap.
    This avoids double-counting nested BE_CHUNK phases inside BE_CHUNKED_BATCH.
    """
    be = [p for p in phases if p["class"] == "BE"]
    top = [p for p in be if p["phase_type"] in TOP_LEVEL_BE_PHASES]
    return top if top else be


def be_work_summary(run_name: str, phases: List[Dict]) -> Optional[Dict]:
    top_be = select_top_level_be_phases(phases)

    if not top_be:
        return None

    durations = [p["duration_ms"] for p in top_be]
    first_begin = min(p["begin_ns"] for p in top_be)
    last_end = max(p["end_ns"] for p in top_be)

    wall_time_ms = (last_end - first_begin) / 1_000_000.0
    total_active_ms = sum(durations)
    aggregate_throughput = len(top_be) / (wall_time_ms / 1000.0) if wall_time_ms > 0 else 0.0

    be_pids = sorted(set(p["pid"] for p in top_be))
    be_processes = len(be_pids)
    per_process_throughput = aggregate_throughput / be_processes if be_processes > 0 else 0.0

    phase_types = sorted(set(p["phase_type"] for p in top_be))

    return {
        "run": run_name,
        "be_phase_types": "+".join(phase_types),
        "be_processes": be_processes,
        "be_phase_count": len(top_be),
        **phase_stats(durations),
        "total_active_ms": total_active_ms,
        "wall_time_ms": wall_time_ms,
        "aggregate_throughput_phases_per_s": aggregate_throughput,
        "per_process_throughput_phases_per_s": per_process_throughput,
    }


# ---------------------------------------------------------------------------
# Overlap analysis
# ---------------------------------------------------------------------------


def overlap_interval_ns(a: Dict, b: Dict) -> Optional[Tuple[int, int]]:
    start = max(a["begin_ns"], b["begin_ns"])
    end = min(a["end_ns"], b["end_ns"])

    if end <= start:
        return None

    return start, end


def interval_ms(interval: Optional[Tuple[int, int]]) -> float:
    if interval is None:
        return 0.0

    start, end = interval
    return (end - start) / 1_000_000.0


def union_interval_ms(intervals: Iterable[Optional[Tuple[int, int]]]) -> float:
    valid = [x for x in intervals if x is not None]

    if not valid:
        return 0.0

    valid.sort()
    merged: List[Tuple[int, int]] = []

    cur_s, cur_e = valid[0]
    for s, e in valid[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e

    merged.append((cur_s, cur_e))
    total_ns = sum(e - s for s, e in merged)
    return total_ns / 1_000_000.0


def max_concurrent_overlaps(intervals: Iterable[Optional[Tuple[int, int]]]) -> int:
    valid = [x for x in intervals if x is not None]

    if not valid:
        return 0

    points: List[Tuple[int, int]] = []
    for s, e in valid:
        points.append((s, 1))
        points.append((e, -1))

    # End before begin on timestamp ties to avoid artificial spikes.
    points.sort(key=lambda x: (x[0], x[1]))

    cur = 0
    peak = 0
    for _, delta in points:
        cur += delta
        peak = max(peak, cur)

    return peak


def generate_overlap_csv(run_name: str, phases: List[Dict], out_dir: Path) -> None:
    lc_phases = [
        p for p in phases
        if p["class"] == "LC" and p["phase_type"] in LC_PHASES_OF_INTEREST
    ]
    be_phases = select_top_level_be_phases(phases)

    if not lc_phases or not be_phases:
        return

    path = out_dir / f"{run_name}_overlap.csv"

    with open(path, "w", newline="") as f:
        fieldnames = [
            "run",
            "lc_pid",
            "lc_tid",
            "lc_phase_id",
            "lc_phase_type",
            "lc_begin_ns",
            "lc_end_ns",
            "lc_duration_ms",
            "be_union_overlap_ms",
            "be_union_overlap_fraction",
            "be_weighted_overlap_ms",
            "be_weighted_overlap_fraction",
            "overlapping_be_phase_count",
            "max_concurrent_be_phases",
        ]

        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for lc in sorted(lc_phases, key=lambda x: x["begin_ns"]):
            intervals = [overlap_interval_ns(lc, be) for be in be_phases]

            union_ms = union_interval_ms(intervals)
            weighted_ms = sum(interval_ms(x) for x in intervals if x is not None)
            count = sum(1 for x in intervals if x is not None)
            max_conc = max_concurrent_overlaps(intervals)
            lc_dur = lc["duration_ms"]

            w.writerow({
                "run": run_name,
                "lc_pid": lc["pid"],
                "lc_tid": lc["tid"],
                "lc_phase_id": lc["phase_id"],
                "lc_phase_type": lc["phase_type"],
                "lc_begin_ns": lc["begin_ns"],
                "lc_end_ns": lc["end_ns"],
                "lc_duration_ms": f"{lc_dur:.6f}",
                "be_union_overlap_ms": f"{union_ms:.6f}",
                "be_union_overlap_fraction": f"{union_ms / lc_dur if lc_dur > 0 else 0.0:.6f}",
                "be_weighted_overlap_ms": f"{weighted_ms:.6f}",
                "be_weighted_overlap_fraction": f"{weighted_ms / lc_dur if lc_dur > 0 else 0.0:.6f}",
                "overlapping_be_phase_count": count,
                "max_concurrent_be_phases": max_conc,
            })

    print(f"[CSV] wrote {path}")


def overlap_summary(run_name: str, phases: List[Dict]) -> Optional[Dict]:
    lc_requests = [
        p for p in phases
        if p["class"] == "LC" and p["phase_type"] == "LC_REQUEST"
    ]
    be_phases = select_top_level_be_phases(phases)

    if not lc_requests or not be_phases:
        return None

    per_request: List[Dict] = []

    for lc in lc_requests:
        intervals = [overlap_interval_ns(lc, be) for be in be_phases]

        union_ms = union_interval_ms(intervals)
        weighted_ms = sum(interval_ms(x) for x in intervals if x is not None)
        count = sum(1 for x in intervals if x is not None)
        max_conc = max_concurrent_overlaps(intervals)

        per_request.append({
            "lc_duration_ms": lc["duration_ms"],
            "be_union_overlap_ms": union_ms,
            "be_weighted_overlap_ms": weighted_ms,
            "be_overlap_count": count,
            "max_concurrent_be": max_conc,
            "be_union_overlap_fraction": union_ms / lc["duration_ms"] if lc["duration_ms"] > 0 else 0.0,
            "be_weighted_overlap_fraction": weighted_ms / lc["duration_ms"] if lc["duration_ms"] > 0 else 0.0,
        })

    durations = [r["lc_duration_ms"] for r in per_request]
    union_overlaps = [r["be_union_overlap_ms"] for r in per_request]
    weighted_overlaps = [r["be_weighted_overlap_ms"] for r in per_request]
    overlap_counts = [r["be_overlap_count"] for r in per_request]
    max_concs = [r["max_concurrent_be"] for r in per_request]
    union_fracs = [r["be_union_overlap_fraction"] for r in per_request]
    weighted_fracs = [r["be_weighted_overlap_fraction"] for r in per_request]

    slow_threshold = percentile(durations, 95)
    slow_rows = [r for r in per_request if r["lc_duration_ms"] >= slow_threshold]
    normal_rows = [r for r in per_request if r["lc_duration_ms"] < slow_threshold]

    def avg(items: List[Dict], key: str) -> float:
        return mean([x[key] for x in items]) if items else 0.0

    return {
        "run": run_name,
        "lc_request_count": len(per_request),
        "lc_p95_ms": percentile(durations, 95),
        "lc_p99_ms": percentile(durations, 99),
        "mean_union_overlap_ms": mean(union_overlaps),
        "p95_union_overlap_ms": percentile(union_overlaps, 95),
        "mean_union_overlap_frac": mean(union_fracs),
        "mean_weighted_overlap_ms": mean(weighted_overlaps),
        "p95_weighted_overlap_ms": percentile(weighted_overlaps, 95),
        "mean_weighted_overlap_frac": mean(weighted_fracs),
        "mean_overlap_count": mean(overlap_counts),
        "p95_overlap_count": percentile(overlap_counts, 95),
        "mean_max_concurrent_be": mean(max_concs),
        "p95_max_concurrent_be": percentile(max_concs, 95),
        "duration_vs_union_overlap_corr": pearson_corr(durations, union_overlaps),
        "duration_vs_weighted_overlap_corr": pearson_corr(durations, weighted_overlaps),
        "slow_request_threshold_ms": slow_threshold,
        "slow_request_count": len(slow_rows),
        "slow_mean_union_overlap_ms": avg(slow_rows, "be_union_overlap_ms"),
        "normal_mean_union_overlap_ms": avg(normal_rows, "be_union_overlap_ms"),
        "slow_mean_weighted_overlap_ms": avg(slow_rows, "be_weighted_overlap_ms"),
        "normal_mean_weighted_overlap_ms": avg(normal_rows, "be_weighted_overlap_ms"),
        "slow_mean_overlap_count": avg(slow_rows, "be_overlap_count"),
        "normal_mean_overlap_count": avg(normal_rows, "be_overlap_count"),
        "slow_mean_max_concurrent_be": avg(slow_rows, "max_concurrent_be"),
        "normal_mean_max_concurrent_be": avg(normal_rows, "max_concurrent_be"),
    }


# ---------------------------------------------------------------------------
# Printing and CSVs
# ---------------------------------------------------------------------------


def repeat_id(run_name: str) -> Optional[int]:
    m = re.search(r"_r(\d+)$", run_name)
    return int(m.group(1)) if m else None


def print_phase_table(rows: List[Dict], title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))

    if not rows:
        print("No rows.")
        return

    header = (
        f"{'class':<6} {'phase_type':<28} {'count':>6} "
        f"{'mean_ms':>10} {'p50_ms':>10} {'p95_ms':>10} "
        f"{'p99_ms':>10} {'max_ms':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['class']:<6} {r['phase_type']:<28} {r['count']:>6} "
            f"{r['mean_ms']:>10.3f} {r['p50_ms']:>10.3f} "
            f"{r['p95_ms']:>10.3f} {r['p99_ms']:>10.3f} "
            f"{r['max_ms']:>10.3f}"
        )


def print_lc_comparison(rows: List[Dict]) -> None:
    print()
    print("#" * 110)
    print("LC REQUEST COMPARISON")
    print("#" * 110)

    if not rows:
        print("No LC rows.")
        return

    baseline_rows = [r for r in rows if r["run"].startswith("lc_alone_stable")]
    if not baseline_rows:
        print("No LC baseline found.")
        return

    base_p95_mean = mean(r["p95_ms"] for r in baseline_rows)
    base_p99_mean = mean(r["p99_ms"] for r in baseline_rows)
    base_p95_median = median(r["p95_ms"] for r in baseline_rows)
    base_p99_median = median(r["p99_ms"] for r in baseline_rows)

    paired_baseline = {
        repeat_id(r["run"]): r
        for r in baseline_rows
        if repeat_id(r["run"]) is not None
    }

    header = (
        f"{'run':<30} {'count':>6} {'mean':>9} {'p50':>9} {'p95':>9} {'p99':>9} {'max':>9} "
        f"{'p95/mean':>10} {'p99/mean':>10} {'p95/med':>10} {'p99/med':>10} "
        f"{'p95/pair':>10} {'p99/pair':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        rid = repeat_id(r["run"])
        paired = paired_baseline.get(rid)

        p95_paired = r["p95_ms"] / paired["p95_ms"] if paired and paired["p95_ms"] > 0 else 0.0
        p99_paired = r["p99_ms"] / paired["p99_ms"] if paired and paired["p99_ms"] > 0 else 0.0

        print(
            f"{r['run']:<30} {r['count']:>6} "
            f"{r['mean_ms']:>9.3f} {r['p50_ms']:>9.3f} {r['p95_ms']:>9.3f} "
            f"{r['p99_ms']:>9.3f} {r['max_ms']:>9.3f} "
            f"{r['p95_ms'] / base_p95_mean:>10.3f} {r['p99_ms'] / base_p99_mean:>10.3f} "
            f"{r['p95_ms'] / base_p95_median:>10.3f} {r['p99_ms'] / base_p99_median:>10.3f} "
            f"{p95_paired:>10.3f} {p99_paired:>10.3f}"
        )

    print()
    print(
        f"LC-alone mean baseline:   P95={base_p95_mean:.3f} ms, P99={base_p99_mean:.3f} ms"
    )
    print(
        f"LC-alone median baseline: P95={base_p95_median:.3f} ms, P99={base_p99_median:.3f} ms"
    )


def print_be_comparison(rows: List[Dict]) -> None:
    print()
    print("#" * 125)
    print("BE WORK COMPARISON")
    print("#" * 125)

    if not rows:
        print("No BE work rows.")
        return

    header = (
        f"{'run':<30} {'be_type':<20} {'BE_procs':>8} {'count':>6} "
        f"{'mean':>9} {'p95':>9} {'p99':>9} "
        f"{'wall_ms':>12} {'agg_thr/s':>12} {'per_proc/s':>12}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['run']:<30} {r['be_phase_types']:<20} "
            f"{r['be_processes']:>8} {r['be_phase_count']:>6} "
            f"{r['mean_ms']:>9.3f} {r['p95_ms']:>9.3f} {r['p99_ms']:>9.3f} "
            f"{r['wall_time_ms']:>12.3f} "
            f"{r['aggregate_throughput_phases_per_s']:>12.3f} "
            f"{r['per_process_throughput_phases_per_s']:>12.3f}"
        )


def print_overlap_comparison(rows: List[Dict]) -> None:
    print()
    print("#" * 150)
    print("LC/BE OVERLAP SUMMARY")
    print("#" * 150)

    if not rows:
        print("No overlap rows.")
        return

    header = (
        f"{'run':<30} {'LC_p95':>8} {'LC_p99':>8} "
        f"{'union_ms':>10} {'weighted_ms':>12} "
        f"{'BE_count':>9} {'max_BE':>8} "
        f"{'corr_u':>8} {'corr_w':>8} "
        f"{'slow_u':>9} {'normal_u':>10} "
        f"{'slow_w':>9} {'normal_w':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['run']:<30} "
            f"{r['lc_p95_ms']:>8.3f} {r['lc_p99_ms']:>8.3f} "
            f"{r['mean_union_overlap_ms']:>10.3f} "
            f"{r['mean_weighted_overlap_ms']:>12.3f} "
            f"{r['mean_overlap_count']:>9.3f} "
            f"{r['mean_max_concurrent_be']:>8.3f} "
            f"{r['duration_vs_union_overlap_corr']:>8.3f} "
            f"{r['duration_vs_weighted_overlap_corr']:>8.3f} "
            f"{r['slow_mean_union_overlap_ms']:>9.3f} "
            f"{r['normal_mean_union_overlap_ms']:>10.3f} "
            f"{r['slow_mean_weighted_overlap_ms']:>9.3f} "
            f"{r['normal_mean_weighted_overlap_ms']:>10.3f}"
        )


def print_condition_readout(lc_rows: List[Dict], be_rows: List[Dict]) -> None:
    print()
    print("#" * 115)
    print("CONDITION READOUT")
    print("#" * 115)

    if not lc_rows:
        print("No LC rows.")
        return

    baseline_rows = [r for r in lc_rows if r["run"].startswith("lc_alone_stable")]
    if not baseline_rows:
        print("No LC baseline rows.")
        return

    base_p95_median = median(r["p95_ms"] for r in baseline_rows)
    base_p99_median = median(r["p99_ms"] for r in baseline_rows)
    be_by_run = {r["run"]: r for r in be_rows}

    header = (
        f"{'run':<30} {'LC_p95':>8} {'LC_p99':>8} "
        f"{'P95x':>8} {'P99x':>8} "
        f"{'BE_procs':>8} {'BE_agg/s':>10} {'BE_proc/s':>10} "
        f"{'readout':<30}"
    )
    print(header)
    print("-" * len(header))

    for lc in lc_rows:
        be = be_by_run.get(lc["run"])
        p95x = lc["p95_ms"] / base_p95_median if base_p95_median > 0 else 0.0
        p99x = lc["p99_ms"] / base_p99_median if base_p99_median > 0 else 0.0

        if lc["run"].startswith("lc_alone"):
            label = "LC baseline"
        elif p95x >= 2.0 or p99x >= 2.0:
            label = "strong LC tail inflation"
        elif p95x >= 1.25 or p99x >= 1.25:
            label = "moderate LC tail inflation"
        elif p95x >= 1.10 or p99x >= 1.10:
            label = "weak LC tail inflation"
        else:
            label = "near baseline"

        be_procs = be["be_processes"] if be else 0
        be_agg = be["aggregate_throughput_phases_per_s"] if be else 0.0
        be_proc = be["per_process_throughput_phases_per_s"] if be else 0.0

        print(
            f"{lc['run']:<30} "
            f"{lc['p95_ms']:>8.3f} {lc['p99_ms']:>8.3f} "
            f"{p95x:>8.3f} {p99x:>8.3f} "
            f"{be_procs:>8} {be_agg:>10.3f} {be_proc:>10.3f} "
            f"{label:<30}"
        )

    print()
    print(
        f"Median LC-alone baseline used here: P95={base_p95_median:.3f} ms, P99={base_p99_median:.3f} ms"
    )


def write_summary_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return

    # Preserve the first row's ordering but include any later extra fields safely.
    fieldnames: List[str] = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[CSV] wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", help="Top-level output dir containing run subdirectories")
    ap.add_argument(
        "--quiet-runs",
        action="store_true",
        help="Do not print each per-run phase table; still prints final summaries.",
    )
    args = ap.parse_args()

    root = Path(args.output_dir)
    if not root.exists():
        raise SystemExit(f"Output directory does not exist: {root}")

    analysis_dir = root / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    run_dirs = sorted([
        p for p in root.iterdir()
        if p.is_dir() and p.name != "analysis"
    ])

    lc_rows: List[Dict] = []
    be_rows: List[Dict] = []
    overlap_rows: List[Dict] = []
    lc_phase_rows: List[Dict] = []

    for run_dir in run_dirs:
        run_name = run_dir.name
        events = collect_events(run_dir)
        phases, unmatched_begins, unmatched_ends, duplicate_begins = pair_phases(events)

        if not args.quiet_runs:
            print()
            print("#" * 100)
            print(f"RUN: {run_name}")
            print("#" * 100)
            print(
                f"events={len(events)} completed_phases={len(phases)} "
                f"unmatched_begins={len(unmatched_begins)} "
                f"unmatched_ends={len(unmatched_ends)} "
                f"duplicate_begins={len(duplicate_begins)}"
            )
            print_phase_table(summarise_by_phase(phases), f"Phase duration summary: {run_name}")

        lc = lc_request_summary(run_name, phases)
        if lc:
            lc_rows.append(lc)

        lc_phase_rows.extend(lc_phase_summary_rows(run_name, phases))

        be = be_work_summary(run_name, phases)
        if be:
            be_rows.append(be)

        if "lc_vs" in run_name:
            generate_overlap_csv(run_name, phases, analysis_dir)
            ov = overlap_summary(run_name, phases)
            if ov:
                overlap_rows.append(ov)

    print_lc_comparison(lc_rows)
    print_be_comparison(be_rows)
    print_overlap_comparison(overlap_rows)
    print_condition_readout(lc_rows, be_rows)

    write_summary_csv(analysis_dir / "lc_request_summary.csv", lc_rows)
    write_summary_csv(analysis_dir / "lc_phase_summary.csv", lc_phase_rows)
    write_summary_csv(analysis_dir / "be_work_summary.csv", be_rows)
    write_summary_csv(analysis_dir / "overlap_summary.csv", overlap_rows)


if __name__ == "__main__":
    main()

