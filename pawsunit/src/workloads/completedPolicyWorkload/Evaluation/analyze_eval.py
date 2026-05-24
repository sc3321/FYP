#!/usr/bin/env python3
"""
analyze_eval.py

Runs parserPolicy.py over every evaluation run root and aggregates the CSVs.

Expected run roots:
  out/<case>_r<repeat>/
    none/
    naive/
    proper/
    analysis/

Usage from Evaluation/:
  python3 analyze_eval.py --out ./out --parser ../parserPolicy.py

Useful flags:
  --no-parse       Do not rerun parserPolicy.py; only aggregate existing CSVs.
  --quiet-runs    Pass --quiet-runs to parserPolicy.py.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics as stats
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


RUN_RE = re.compile(r"^(?P<case>.+)_r(?P<repeat>\d+)$")


def infer_case_repeat(path: Path) -> Tuple[str, Optional[int]]:
    m = RUN_RE.match(path.name)
    if not m:
        return path.name, None
    return m.group("case"), int(m.group("repeat"))


def has_policy_dirs(path: Path) -> bool:
    return any((path / p).is_dir() for p in ("none", "naive", "proper"))


def discover_run_roots(out: Path) -> List[Path]:
    roots = []
    for child in sorted(out.iterdir()):
        if child.is_dir() and has_policy_dirs(child):
            roots.append(child)
    return roots


def run_parser(parser: Path, root: Path, quiet_runs: bool) -> None:
    cmd = [sys.executable, str(parser), str(root)]
    if quiet_runs:
        cmd.append("--quiet-runs")
    print(f"[PARSE] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[SKIP] no rows for {path}")
        return

    # Preserve common metadata first, then all other fields.
    fields = []
    for preferred in ("case", "repeat", "policy", "run"):
        if preferred in rows[0] and preferred not in fields:
            fields.append(preferred)

    seen = set(fields)
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"[CSV] wrote {path}")


def augment_rows(rows: List[Dict[str, str]], case: str, repeat: Optional[int]) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        new = dict(row)
        new["case"] = case
        new["repeat"] = "" if repeat is None else str(repeat)
        out.append(new)
    return out


def first_existing(row: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in row and row[c] not in ("", None):
            return c
    return None


def to_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def group_summary(rows: List[Dict[str, str]], metric_candidates: Dict[str, List[str]]) -> List[Dict[str, str]]:
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        case = row.get("case", "UNK")
        policy = row.get("policy", row.get("run", "UNK"))
        groups.setdefault((case, policy), []).append(row)

    output: List[Dict[str, str]] = []
    for (case, policy), grows in sorted(groups.items()):
        result: Dict[str, str] = {
            "case": case,
            "policy": policy,
            "n": str(len(grows)),
        }

        for metric_name, candidates in metric_candidates.items():
            vals = []
            used_col = None
            for row in grows:
                col = first_existing(row, candidates)
                if col is None:
                    continue
                used_col = col
                val = to_float(row[col])
                if val is not None:
                    vals.append(val)

            if vals:
                result[f"{metric_name}_mean"] = f"{stats.mean(vals):.6f}"
                result[f"{metric_name}_median"] = f"{stats.median(vals):.6f}"
                result[f"{metric_name}_min"] = f"{min(vals):.6f}"
                result[f"{metric_name}_max"] = f"{max(vals):.6f}"
                if len(vals) >= 2:
                    result[f"{metric_name}_stdev"] = f"{stats.stdev(vals):.6f}"
                else:
                    result[f"{metric_name}_stdev"] = "0.000000"
                result[f"{metric_name}_source_col"] = used_col or ""
        output.append(result)

    return output


def print_compact(title: str, rows: List[Dict[str, str]], metrics: List[str]) -> None:
    if not rows:
        return

    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    headers = ["case", "policy", "n"] + [f"{m}_mean" for m in metrics]
    widths = {h: max(len(h), 12) for h in headers}

    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(row.get(h, "")))

    print(" ".join(h.ljust(widths[h]) for h in headers))
    print("-" * sum(widths.values()))

    for row in rows:
        print(" ".join(row.get(h, "").ljust(widths[h]) for h in headers))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./out", help="Evaluation output root")
    ap.add_argument("--parser", default="../parserPolicy.py", help="Path to parserPolicy.py")
    ap.add_argument("--no-parse", action="store_true", help="Only aggregate existing analysis CSVs")
    ap.add_argument("--quiet-runs", action="store_true", default=True, help="Pass --quiet-runs to parserPolicy.py")
    args = ap.parse_args()

    out = Path(args.out)
    parser = Path(args.parser)

    if not out.exists():
        raise SystemExit(f"[ERROR] output root not found: {out}")

    roots = discover_run_roots(out)
    if not roots:
        raise SystemExit(f"[ERROR] no run roots found under {out}. Expected out/<case>_r<repeat>/none|naive|proper")

    print(f"[INFO] discovered {len(roots)} run roots")

    if not args.no_parse:
        if not parser.exists():
            raise SystemExit(f"[ERROR] parser not found: {parser}")
        for root in roots:
            run_parser(parser, root, args.quiet_runs)

    all_lc: List[Dict[str, str]] = []
    all_be: List[Dict[str, str]] = []
    all_overlap: List[Dict[str, str]] = []
    all_lc_phase: List[Dict[str, str]] = []

    for root in roots:
        case, repeat = infer_case_repeat(root)
        analysis = root / "analysis"

        all_lc.extend(augment_rows(read_csv(analysis / "lc_request_summary.csv"), case, repeat))
        all_lc_phase.extend(augment_rows(read_csv(analysis / "lc_phase_summary.csv"), case, repeat))
        all_be.extend(augment_rows(read_csv(analysis / "be_work_summary.csv"), case, repeat))
        all_overlap.extend(augment_rows(read_csv(analysis / "overlap_summary.csv"), case, repeat))

    combined = out / "analysis_all"
    write_csv(combined / "lc_request_summary_all.csv", all_lc)
    write_csv(combined / "lc_phase_summary_all.csv", all_lc_phase)
    write_csv(combined / "be_work_summary_all.csv", all_be)
    write_csv(combined / "overlap_summary_all.csv", all_overlap)

    lc_metrics = {
        "lc_mean_ms": ["mean_ms", "mean"],
        "lc_p50_ms": ["p50_ms", "p50"],
        "lc_p95_ms": ["p95_ms", "lc_p95_ms", "LC_p95", "p95"],
        "lc_p99_ms": ["p99_ms", "lc_p99_ms", "LC_p99", "p99"],
        "lc_max_ms": ["max_ms", "max"],
    }

    be_metrics = {
        "be_mean_ms": ["mean_ms", "mean"],
        "be_p50_ms": ["p50_ms", "p50"],
        "be_p95_ms": ["p95_ms", "p95"],
        "be_p99_ms": ["p99_ms", "p99"],
        "be_max_ms": ["max_ms", "max"],
        "be_total_active_ms": ["total_active_ms"],
        "be_wall_ms": ["wall_time_ms", "wall_ms"],
        "be_agg_thr_s": [
            "aggregate_throughput_phases_per_s",
            "throughput_phases_per_s",
            "agg_thr_per_s",
            "agg_thr/s",
            "BE_agg/s",
        ],
        "be_per_proc_s": [
            "per_process_throughput_phases_per_s",
            "per_proc_s",
            "per_proc/s",
        ],
    }

    overlap_metrics = {
        "union_ms": ["mean_union_overlap_ms", "union_ms"],
        "weighted_ms": ["mean_weighted_overlap_ms", "weighted_ms"],
        "overlap_count": ["mean_overlap_count", "BE_count"],
        "max_be": ["mean_max_concurrent_be", "max_BE"],
    }

    lc_summary = group_summary(all_lc, lc_metrics)
    be_summary = group_summary(all_be, be_metrics)
    overlap_summary = group_summary(all_overlap, overlap_metrics)

    write_csv(combined / "lc_request_grouped.csv", lc_summary)
    write_csv(combined / "be_work_grouped.csv", be_summary)
    write_csv(combined / "overlap_grouped.csv", overlap_summary)

    print_compact("LC REQUEST GROUPED SUMMARY", lc_summary, ["lc_p95_ms", "lc_p99_ms"])
    print_compact("BE WORK GROUPED SUMMARY", be_summary, ["be_agg_thr_s", "be_wall_ms"])
    print_compact("OVERLAP GROUPED SUMMARY", overlap_summary, ["union_ms", "weighted_ms", "overlap_count"])


if __name__ == "__main__":
    main()

