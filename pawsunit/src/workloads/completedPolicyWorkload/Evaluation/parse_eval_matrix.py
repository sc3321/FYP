#!/usr/bin/env python3
"""
parse_eval_matrix.py

Parses Phase 3 policy evaluation output directories produced by runner_eval_matrix.sh.

Expected modern layout:

  OUT_ROOT/
    lc_vs_mixed/
      mixed4/
        proper/
          delay_1000/
            r1/
              config.txt
              lc_events/...
              be1_events/...
              be2_events/...
              ...

Also supports older shallow layouts where each direct child of OUT_ROOT is a run.

Outputs under OUT_ROOT/analysis/:

  per_run_summary.csv
  lc_request_per_run.csv
  lc_phase_per_run.csv
  be_work_per_run.csv
  overlap_per_run.csv
  grouped_lc_request.csv
  grouped_be_work.csv
  grouped_overlap.csv
  grouped_tradeoff.csv
  per_request_overlap/*.csv

Main metrics:
  - LC p50/p95/p99/max/mean per run and grouped across repeats
  - BE aggregate throughput and wall time per run and grouped across repeats
  - LC/BE overlap union, weighted overlap, overlap counts, max concurrent BE
  - delay-sweep friendly tradeoff table
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, Iterable, List, Optional, Tuple, Any

EVENT_RE = re.compile(
    r"Event type = (?P<event>BEGIN|END): "
    r"PhaseId:\[(?P<pid>\d+), (?P<phase_id>\d+)\],"
    r"Thread Id: (?P<tid>\d+), "
    r"parent_id: (?P<parent_id>-?\d+), "
    r"depth: (?P<depth>\d+),\s+"
    r"Timestamp: (?P<sec>\d+) s (?P<nsec>\d+) ns, "
    r"phase type: \((?P<phase_type>[^)]*)\), "
    r"workload class: (?P<class>\w+)"
    r"(?:, granularity: (?P<granularity>\w+))?"
)

TOP_LEVEL_BE_PHASES = {"BE_LONG_BATCH", "BE_CHUNKED_BATCH"}
LC_PHASES_OF_INTEREST = {
    "LC_REQUEST",
    "LC_PREFILL_SYNC",
    "LC_DECODE_STEP",
    "LC_PREFILL_SUBMISSION",
}

NUMERIC_GROUP_FIELDS = [
    "count",
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "max_ms",
    "min_ms",
    "stdev_ms",
    "total_active_ms",
    "wall_time_ms",
    "aggregate_throughput_phases_per_s",
    "per_process_throughput_phases_per_s",
    "mean_union_overlap_ms",
    "p95_union_overlap_ms",
    "mean_union_overlap_frac",
    "mean_weighted_overlap_ms",
    "p95_weighted_overlap_ms",
    "mean_weighted_overlap_frac",
    "mean_overlap_count",
    "p95_overlap_count",
    "mean_max_concurrent_be",
    "p95_max_concurrent_be",
    "duration_vs_union_overlap_corr",
    "duration_vs_weighted_overlap_corr",
    "slow_request_threshold_ms",
    "slow_request_count",
    "slow_mean_union_overlap_ms",
    "normal_mean_union_overlap_ms",
    "slow_mean_weighted_overlap_ms",
    "normal_mean_weighted_overlap_ms",
    "slow_mean_overlap_count",
    "normal_mean_overlap_count",
    "slow_mean_max_concurrent_be",
    "normal_mean_max_concurrent_be",
]

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


def safe_mean(vals: List[float]) -> float:
    return mean(vals) if vals else 0.0


def safe_median(vals: List[float]) -> float:
    return median(vals) if vals else 0.0


def safe_stdev(vals: List[float]) -> float:
    return stdev(vals) if len(vals) > 1 else 0.0


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
        "stdev_ms": safe_stdev(vals),
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
# Config and run discovery
# ---------------------------------------------------------------------------


def parse_config(path: Path) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    if not path.exists():
        return cfg

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()

    return cfg


def int_from_cfg(cfg: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(cfg.get(key, default))
    except ValueError:
        return default


def str_from_cfg(cfg: Dict[str, str], key: str, default: str = "") -> str:
    return cfg.get(key, default)


def discover_run_dirs(root: Path) -> List[Path]:
    """Find actual run dirs. Prefer directories containing config.txt."""
    config_dirs = sorted(p.parent for p in root.rglob("config.txt") if "analysis" not in p.parts)
    if config_dirs:
        return config_dirs

    # Fallback for older layout: direct children are runs.
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name != "analysis"])


def metadata_from_run_dir(root: Path, run_dir: Path) -> Dict[str, Any]:
    cfg = parse_config(run_dir / "config.txt")

    rel_parts = run_dir.relative_to(root).parts

    # Defaults from config first.
    case = str_from_cfg(cfg, "CASE", "")
    wcfg = str_from_cfg(cfg, "WORKER_CONFIG", "")
    policy = str_from_cfg(cfg, "POLICY", "")
    delay_us = int_from_cfg(cfg, "BE_DELAY_US", 0)
    repeat = int_from_cfg(cfg, "REPEAT", 0)

    # Infer from modern path if missing.
    # case/wcfg/policy/delay_X/rY
    if len(rel_parts) >= 5:
        case = case or rel_parts[-5]
        wcfg = wcfg or rel_parts[-4]
        policy = policy or rel_parts[-3]

        m_delay = re.match(r"delay_(\d+)", rel_parts[-2])
        if delay_us == 0 and m_delay:
            delay_us = int(m_delay.group(1))

        m_repeat = re.match(r"r(\d+)", rel_parts[-1])
        if repeat == 0 and m_repeat:
            repeat = int(m_repeat.group(1))

    # Infer old style names if still missing.
    if not case:
        case = rel_parts[0] if rel_parts else run_dir.name
    if not policy:
        policy = rel_parts[-1] if rel_parts else "unknown"

    long_be_workers = int_from_cfg(cfg, "LONG_BE_WORKERS", 0)
    chunked_be_workers = int_from_cfg(cfg, "CHUNKED_BE_WORKERS", 0)
    total_be_workers = int_from_cfg(cfg, "TOTAL_BE_WORKERS", long_be_workers + chunked_be_workers)

    # Old runner compatibility.
    if total_be_workers == 0:
        old_be_workers = int_from_cfg(cfg, "BE_WORKERS", 0)
        old_long = int_from_cfg(cfg, "BE_LONG_WORKERS", 0)
        old_chunked = int_from_cfg(cfg, "BE_CHUNKED_WORKERS", 0)
        long_be_workers = old_long
        chunked_be_workers = old_chunked
        total_be_workers = old_long + old_chunked + old_be_workers

    run_label = f"{case}/{wcfg or 'unknown_wcfg'}/{policy}/delay_{delay_us}/r{repeat}"

    return {
        "run_label": run_label,
        "run_dir": str(run_dir),
        "case": case,
        "worker_config": wcfg or "unknown",
        "policy": policy,
        "delay_us": delay_us,
        "repeat": repeat,
        "long_be_workers": long_be_workers,
        "chunked_be_workers": chunked_be_workers,
        "total_be_workers": total_be_workers,
        "start_order": str_from_cfg(cfg, "START_ORDER", "unknown"),
        "stagger_sec": str_from_cfg(cfg, "STAGGER_SEC", ""),
        "lc_iters": int_from_cfg(cfg, "LC_ITERS", 0),
        "be_iters": int_from_cfg(cfg, "BE_ITERS", 0),
        "chunks": int_from_cfg(cfg, "CHUNKS", 0),
        "lc_inner": int_from_cfg(cfg, "LC_INNER", 0),
        "be_inner": int_from_cfg(cfg, "BE_INNER", 0),
        "n": int_from_cfg(cfg, "N", 0),
    }


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------


def is_event_log_path(path: Path) -> bool:
    parts = set(path.parts)

    if "events" in parts or "lc_events" in parts or "be_events" in parts:
        return True

    return any(re.fullmatch(r"be\d+_events", part) for part in path.parts)


def parse_event_file(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

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


def collect_events(run_dir: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    for path in run_dir.rglob("*"):
        if path.is_file() and is_event_log_path(path):
            events.extend(parse_event_file(path))

    return events


def pair_phases(events: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    begins: Dict[Tuple[int, int], Dict[str, Any]] = {}
    completed: List[Dict[str, Any]] = []
    unmatched_ends: List[Dict[str, Any]] = []
    duplicate_begins: List[Dict[str, Any]] = []

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
                "granularity": b["granularity"],
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


def add_meta(row: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    return {**meta, **row}


def summarise_by_phase(phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str], List[float]] = {}

    for p in phases:
        key = (p["class"], p["phase_type"])
        by_key.setdefault(key, []).append(p["duration_ms"])

    rows: List[Dict[str, Any]] = []
    for (klass, phase_type), vals in sorted(by_key.items()):
        rows.append({"class": klass, "phase_type": phase_type, **phase_stats(vals)})

    return rows


def lc_request_summary(phases: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    vals = [
        p["duration_ms"]
        for p in phases
        if p["class"] == "LC" and p["phase_type"] == "LC_REQUEST"
    ]

    if not vals:
        return None

    return phase_stats(vals)


def lc_phase_summary_rows(phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for phase_type in sorted(LC_PHASES_OF_INTEREST):
        vals = [
            p["duration_ms"]
            for p in phases
            if p["class"] == "LC" and p["phase_type"] == phase_type
        ]
        if vals:
            rows.append({"phase_type": phase_type, **phase_stats(vals)})

    return rows


def select_top_level_be_phases(phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prefer top-level BE_LONG_BATCH / BE_CHUNKED_BATCH for throughput and overlap.
    This avoids double-counting nested BE_CHUNK phases inside BE_CHUNKED_BATCH.
    """
    be = [p for p in phases if p["class"] == "BE"]
    top = [p for p in be if p["phase_type"] in TOP_LEVEL_BE_PHASES]
    return top if top else be


def be_work_summary(phases: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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
    granularities = sorted(set(p.get("granularity", "UNK") for p in top_be))

    return {
        "be_phase_types": "+".join(phase_types),
        "be_granularities": "+".join(granularities),
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


def overlap_interval_ns(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[Tuple[int, int]]:
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


def per_request_overlap_rows(phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lc_requests = [
        p for p in phases
        if p["class"] == "LC" and p["phase_type"] == "LC_REQUEST"
    ]
    be_phases = select_top_level_be_phases(phases)

    rows: List[Dict[str, Any]] = []

    if not lc_requests or not be_phases:
        return rows

    for lc in sorted(lc_requests, key=lambda x: x["begin_ns"]):
        intervals = [overlap_interval_ns(lc, be) for be in be_phases]

        union_ms = union_interval_ms(intervals)
        weighted_ms = sum(interval_ms(x) for x in intervals if x is not None)
        count = sum(1 for x in intervals if x is not None)
        max_conc = max_concurrent_overlaps(intervals)
        lc_dur = lc["duration_ms"]

        rows.append({
            "lc_pid": lc["pid"],
            "lc_tid": lc["tid"],
            "lc_phase_id": lc["phase_id"],
            "lc_begin_ns": lc["begin_ns"],
            "lc_end_ns": lc["end_ns"],
            "lc_duration_ms": lc_dur,
            "be_union_overlap_ms": union_ms,
            "be_union_overlap_fraction": union_ms / lc_dur if lc_dur > 0 else 0.0,
            "be_weighted_overlap_ms": weighted_ms,
            "be_weighted_overlap_fraction": weighted_ms / lc_dur if lc_dur > 0 else 0.0,
            "overlapping_be_phase_count": count,
            "max_concurrent_be_phases": max_conc,
        })

    return rows


def overlap_summary(phases: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    per_request = per_request_overlap_rows(phases)

    if not per_request:
        return None

    durations = [r["lc_duration_ms"] for r in per_request]
    union_overlaps = [r["be_union_overlap_ms"] for r in per_request]
    weighted_overlaps = [r["be_weighted_overlap_ms"] for r in per_request]
    overlap_counts = [r["overlapping_be_phase_count"] for r in per_request]
    max_concs = [r["max_concurrent_be_phases"] for r in per_request]
    union_fracs = [r["be_union_overlap_fraction"] for r in per_request]
    weighted_fracs = [r["be_weighted_overlap_fraction"] for r in per_request]

    slow_threshold = percentile(durations, 95)
    slow_rows = [r for r in per_request if r["lc_duration_ms"] >= slow_threshold]
    normal_rows = [r for r in per_request if r["lc_duration_ms"] < slow_threshold]

    def avg(items: List[Dict[str, Any]], key: str) -> float:
        return mean([x[key] for x in items]) if items else 0.0

    return {
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
        "slow_mean_overlap_count": avg(slow_rows, "overlapping_be_phase_count"),
        "normal_mean_overlap_count": avg(normal_rows, "overlapping_be_phase_count"),
        "slow_mean_max_concurrent_be": avg(slow_rows, "max_concurrent_be_phases"),
        "normal_mean_max_concurrent_be": avg(normal_rows, "max_concurrent_be_phases"),
    }


# ---------------------------------------------------------------------------
# Grouped summaries
# ---------------------------------------------------------------------------


def group_key(row: Dict[str, Any], keys: List[str]) -> Tuple[Any, ...]:
    return tuple(row.get(k, "") for k in keys)


def grouped_mean_rows(rows: List[Dict[str, Any]], keys: List[str], numeric_fields: List[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}

    for row in rows:
        buckets.setdefault(group_key(row, keys), []).append(row)

    out: List[Dict[str, Any]] = []

    for key_tuple, bucket in sorted(buckets.items(), key=lambda kv: kv[0]):
        result: Dict[str, Any] = {k: v for k, v in zip(keys, key_tuple)}
        result["n"] = len(bucket)

        for field in numeric_fields:
            vals: List[float] = []
            for row in bucket:
                value = row.get(field)
                if value is None or value == "":
                    continue
                try:
                    vals.append(float(value))
                except (TypeError, ValueError):
                    continue

            if not vals:
                continue

            result[f"{field}_mean"] = mean(vals)
            result[f"{field}_median"] = median(vals)
            result[f"{field}_stdev"] = safe_stdev(vals)
            result[f"{field}_min"] = min(vals)
            result[f"{field}_max"] = max(vals)

        out.append(result)

    return out


def make_tradeoff_rows(grouped_lc: List[Dict[str, Any]], grouped_be: List[Dict[str, Any]], grouped_ov: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = ["case", "worker_config", "policy", "delay_us", "total_be_workers", "long_be_workers", "chunked_be_workers", "start_order"]

    def idx(rows: List[Dict[str, Any]]) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
        return {group_key(r, keys): r for r in rows}

    be_idx = idx(grouped_be)
    ov_idx = idx(grouped_ov)

    out: List[Dict[str, Any]] = []

    for lc in grouped_lc:
        k = group_key(lc, keys)
        be = be_idx.get(k, {})
        ov = ov_idx.get(k, {})

        row: Dict[str, Any] = {key: lc.get(key, "") for key in keys}
        row["n"] = lc.get("n", 0)

        row["lc_p50_ms_mean"] = lc.get("p50_ms_mean", 0.0)
        row["lc_p95_ms_mean"] = lc.get("p95_ms_mean", 0.0)
        row["lc_p99_ms_mean"] = lc.get("p99_ms_mean", 0.0)
        row["lc_max_ms_mean"] = lc.get("max_ms_mean", 0.0)

        row["be_agg_thr_s_mean"] = be.get("aggregate_throughput_phases_per_s_mean", 0.0)
        row["be_per_proc_thr_s_mean"] = be.get("per_process_throughput_phases_per_s_mean", 0.0)
        row["be_wall_ms_mean"] = be.get("wall_time_ms_mean", 0.0)
        row["be_total_active_ms_mean"] = be.get("total_active_ms_mean", 0.0)
        row["be_phase_count_mean"] = be.get("be_phase_count_mean", 0.0)

        row["overlap_union_ms_mean"] = ov.get("mean_union_overlap_ms_mean", 0.0)
        row["overlap_weighted_ms_mean"] = ov.get("mean_weighted_overlap_ms_mean", 0.0)
        row["overlap_count_mean"] = ov.get("mean_overlap_count_mean", 0.0)
        row["max_concurrent_be_mean"] = ov.get("mean_max_concurrent_be_mean", 0.0)

        out.append(row)

    return sorted(out, key=lambda r: (
        str(r.get("case", "")),
        str(r.get("worker_config", "")),
        str(r.get("policy", "")),
        int(r.get("delay_us", 0) or 0),
    ))


# ---------------------------------------------------------------------------
# CSV + printing
# ---------------------------------------------------------------------------


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[CSV] wrote {path}")


def print_table(rows: List[Dict[str, Any]], title: str, cols: List[Tuple[str, str, int, str]]) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))

    if not rows:
        print("No rows.")
        return

    header = " ".join(f"{name:<{width}}" if align == "<" else f"{name:>{width}}" for key, name, width, align in cols)
    print(header)
    print("-" * len(header))

    for row in rows:
        parts = []
        for key, name, width, align in cols:
            value = row.get(key, "")
            if isinstance(value, float):
                value_s = f"{value:.6f}"
            else:
                value_s = str(value)
            if align == "<":
                parts.append(f"{value_s:<{width}}")
            else:
                parts.append(f"{value_s:>{width}}")
        print(" ".join(parts))


def print_grouped_readout(tradeoff_rows: List[Dict[str, Any]]) -> None:
    cols = [
        ("case", "case", 18, "<"),
        ("worker_config", "workers", 10, "<"),
        ("policy", "policy", 8, "<"),
        ("delay_us", "delay", 8, ">"),
        ("n", "n", 4, ">"),
        ("lc_p95_ms_mean", "lc_p95", 10, ">"),
        ("lc_p99_ms_mean", "lc_p99", 10, ">"),
        ("be_agg_thr_s_mean", "be_thr/s", 12, ">"),
        ("be_wall_ms_mean", "be_wall", 12, ">"),
        ("overlap_union_ms_mean", "union_ms", 10, ">"),
        ("overlap_weighted_ms_mean", "weighted_ms", 12, ">"),
        ("overlap_count_mean", "ov_count", 10, ">"),
    ]
    print_table(tradeoff_rows, "GROUPED TRADEOFF SUMMARY", cols)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", help="Top-level output dir containing matrix output")
    ap.add_argument("--quiet-runs", action="store_true", help="Suppress per-run console diagnostics")
    ap.add_argument("--write-per-request", action="store_true", help="Write one per-request overlap CSV per run")
    args = ap.parse_args()

    root = Path(args.output_dir)
    if not root.exists():
        raise SystemExit(f"Output directory does not exist: {root}")

    analysis_dir = root / "analysis"
    per_request_dir = analysis_dir / "per_request_overlap"
    analysis_dir.mkdir(exist_ok=True)

    run_dirs = discover_run_dirs(root)
    if not run_dirs:
        raise SystemExit(f"No run directories found under: {root}")

    per_run_rows: List[Dict[str, Any]] = []
    lc_rows: List[Dict[str, Any]] = []
    lc_phase_rows: List[Dict[str, Any]] = []
    be_rows: List[Dict[str, Any]] = []
    overlap_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        meta = metadata_from_run_dir(root, run_dir)
        events = collect_events(run_dir)
        phases, unmatched_begins, unmatched_ends, duplicate_begins = pair_phases(events)

        diagnostics = {
            "event_count": len(events),
            "completed_phase_count": len(phases),
            "unmatched_begin_count": len(unmatched_begins),
            "unmatched_end_count": len(unmatched_ends),
            "duplicate_begin_count": len(duplicate_begins),
        }

        per_run_rows.append({**meta, **diagnostics})

        if not args.quiet_runs:
            print(
                f"[RUN] {meta['run_label']} events={len(events)} phases={len(phases)} "
                f"unmatched_begin={len(unmatched_begins)} unmatched_end={len(unmatched_ends)} dup_begin={len(duplicate_begins)}"
            )

        lc = lc_request_summary(phases)
        if lc:
            lc_rows.append(add_meta(lc, meta))

        for row in lc_phase_summary_rows(phases):
            lc_phase_rows.append(add_meta(row, meta))

        be = be_work_summary(phases)
        if be:
            be_rows.append(add_meta(be, meta))

        ov = overlap_summary(phases)
        if ov:
            overlap_rows.append(add_meta(ov, meta))

        if args.write_per_request:
            req_rows = per_request_overlap_rows(phases)
            if req_rows:
                safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "__", meta["run_label"])
                write_summary_csv(per_request_dir / f"{safe_label}.csv", [add_meta(r, meta) for r in req_rows])

    group_keys = [
        "case",
        "worker_config",
        "policy",
        "delay_us",
        "total_be_workers",
        "long_be_workers",
        "chunked_be_workers",
        "start_order",
    ]

    lc_numeric = ["count", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "min_ms", "stdev_ms"]
    be_numeric = [
        "be_processes",
        "be_phase_count",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "min_ms",
        "stdev_ms",
        "total_active_ms",
        "wall_time_ms",
        "aggregate_throughput_phases_per_s",
        "per_process_throughput_phases_per_s",
    ]
    ov_numeric = [
        "lc_request_count",
        "lc_p95_ms",
        "lc_p99_ms",
        "mean_union_overlap_ms",
        "p95_union_overlap_ms",
        "mean_union_overlap_frac",
        "mean_weighted_overlap_ms",
        "p95_weighted_overlap_ms",
        "mean_weighted_overlap_frac",
        "mean_overlap_count",
        "p95_overlap_count",
        "mean_max_concurrent_be",
        "p95_max_concurrent_be",
        "duration_vs_union_overlap_corr",
        "duration_vs_weighted_overlap_corr",
        "slow_request_threshold_ms",
        "slow_request_count",
        "slow_mean_union_overlap_ms",
        "normal_mean_union_overlap_ms",
        "slow_mean_weighted_overlap_ms",
        "normal_mean_weighted_overlap_ms",
        "slow_mean_overlap_count",
        "normal_mean_overlap_count",
        "slow_mean_max_concurrent_be",
        "normal_mean_max_concurrent_be",
    ]

    grouped_lc = grouped_mean_rows(lc_rows, group_keys, lc_numeric)
    grouped_be = grouped_mean_rows(be_rows, group_keys, be_numeric)
    grouped_ov = grouped_mean_rows(overlap_rows, group_keys, ov_numeric)
    tradeoff_rows = make_tradeoff_rows(grouped_lc, grouped_be, grouped_ov)

    print_grouped_readout(tradeoff_rows)

    write_summary_csv(analysis_dir / "per_run_summary.csv", per_run_rows)
    write_summary_csv(analysis_dir / "lc_request_per_run.csv", lc_rows)
    write_summary_csv(analysis_dir / "lc_phase_per_run.csv", lc_phase_rows)
    write_summary_csv(analysis_dir / "be_work_per_run.csv", be_rows)
    write_summary_csv(analysis_dir / "overlap_per_run.csv", overlap_rows)

    write_summary_csv(analysis_dir / "grouped_lc_request.csv", grouped_lc)
    write_summary_csv(analysis_dir / "grouped_be_work.csv", grouped_be)
    write_summary_csv(analysis_dir / "grouped_overlap.csv", grouped_ov)
    write_summary_csv(analysis_dir / "grouped_tradeoff.csv", tradeoff_rows)

    print()
    print(f"[DONE] Parsed {len(run_dirs)} runs.")
    print(f"[DONE] Analysis written under: {analysis_dir}")


if __name__ == "__main__":
    main()

