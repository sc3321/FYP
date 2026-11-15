#!/usr/bin/env python3
"""
analyse_dvfs_runs.py

Parses the 20 DVFS log runs in dvfs/runs/ and the corresponding llama
latency results in llama_validation/runs/, then produces:

  dvfs/analysis/dvfs_summary.csv         -- per-case DVFS behaviour
  dvfs/analysis/llama_dvfs_summary.csv   -- LC/BE latency per case, with DVFS

Usage:
  python3 analyse_dvfs_runs.py [dvfs_runs_dir] [llama_runs_dir]

Defaults:
  dvfs_runs_dir  = ~/FYP/dvfs/runs
  llama_runs_dir = ~/FYP/llama_validation/runs

What each output file contains
-------------------------------
dvfs_summary.csv
  One row per (run, case). Columns:
    run_dir, case_name, policy, n_samples,
    cap_binding_frac      -- fraction of samples where powerDrawMw > limitMw * 0.85
                             (proxy for whether the cap was actually doing anything)
    mean_smClock_mhz      -- average achieved SM clock over the case
    lc_mean_clock_mhz     -- mean clock when activeLC > 0
    be_only_mean_clock_mhz-- mean clock when activeLC == 0 and activeBELong > 0
    idle_mean_clock_mhz   -- mean clock when all counters == 0
    n_actuations          -- number of times the daemon actually changed the limit
    lc_mean_limit_mw      -- mean power limit when LC is active
    be_only_mean_limit_mw -- mean power limit when BE-only
    idle_mean_limit_mw    -- mean power limit when idle
    mean_temp_c           -- mean GPU temperature

llama_dvfs_summary.csv
  One row per (run, case). Columns:
    run_dir, case_name, policy,
    lc_client_p50_ms, lc_client_p95_ms, lc_client_p99_ms,
    be_client_p50_ms, be_client_p95_ms, be_client_p99_ms,
    n_lc_requests, n_be_requests

Both files can be opened directly in Excel/LibreOffice or read by a plotting
script. Rows with missing data are included with empty cells rather than
omitted, so you can tell what ran and what was incomplete.
"""

import csv
import json
import math
import statistics
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(math.ceil(p / 100.0 * len(xs)) - 1, len(xs) - 1))
    return xs[k]


def safe_mean(xs):
    return statistics.mean(xs) if xs else None


def fmt(x, dp=1):
    if x is None:
        return ""
    return f"{x:.{dp}f}"


# ---------------------------------------------------------------------------
# dvfs_log.csv parsing
# ---------------------------------------------------------------------------
# Columns written by dvfsManager::logSample:
#   wall_ms, activeLC, activeBELong, activeBEChunked,
#   targetMw, limitMw, changed, powerDrawMw, smClockMHz, tempC

def parse_dvfs_log(path: Path):
    """Return list of row dicts from a dvfs_log.csv."""
    rows = []
    try:
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({
                        "wall_ms":        int(row["wall_ms"]),
                        "activeLC":       int(row["activeLC"]),
                        "activeBELong":   int(row["activeBELong"]),
                        "activeBEChunked":int(row["activeBEChunked"]),
                        "targetMw":       int(row["targetMw"]),
                        "limitMw":        int(row["limitMw"]),
                        "changed":        int(row["changed"]),
                        "powerDrawMw":    int(row["powerDrawMw"]),
                        "smClockMHz":     int(row["smClockMHz"]),
                        "tempC":          int(row["tempC"]),
                    })
                except (KeyError, ValueError):
                    continue
    except FileNotFoundError:
        pass
    return rows


def summarise_dvfs(rows):
    """Compute per-case DVFS metrics from a parsed dvfs_log."""
    if not rows:
        return {}

    # Validity filter: drop rows with implausible clock (0 or >3000 MHz)
    rows = [r for r in rows if 0 < r["smClockMHz"] < 3000]
    if not rows:
        return {}

    # Partition rows by workload state
    lc_rows     = [r for r in rows if r["activeLC"] > 0]
    be_only_rows= [r for r in rows if r["activeLC"] == 0 and r["activeBELong"] > 0]
    idle_rows   = [r for r in rows if r["activeLC"] == 0 and r["activeBELong"] == 0
                                   and r["activeBEChunked"] == 0]

    # Cap binding: draw > 85% of limit means the cap is constraining Boost
    cap_binding = [r for r in rows if r["limitMw"] > 0
                   and r["powerDrawMw"] > 0.85 * r["limitMw"]]

    actuations = sum(r["changed"] for r in rows)

    return {
        "n_samples":              len(rows),
        "cap_binding_frac":       len(cap_binding) / len(rows) if rows else None,
        "mean_smClock_mhz":       safe_mean([r["smClockMHz"] for r in rows]),
        "lc_mean_clock_mhz":      safe_mean([r["smClockMHz"] for r in lc_rows]),
        "be_only_mean_clock_mhz": safe_mean([r["smClockMHz"] for r in be_only_rows]),
        "idle_mean_clock_mhz":    safe_mean([r["smClockMHz"] for r in idle_rows]),
        "n_actuations":           actuations,
        "lc_mean_limit_mw":       safe_mean([r["limitMw"] for r in lc_rows]),
        "be_only_mean_limit_mw":  safe_mean([r["limitMw"] for r in be_only_rows]),
        "idle_mean_limit_mw":     safe_mean([r["limitMw"] for r in idle_rows]),
        "mean_temp_c":            safe_mean([r["tempC"] for r in rows]),
    }


# ---------------------------------------------------------------------------
# llama_validation/runs parsing  (reuses lc_client.jsonl / be_client.jsonl)
# ---------------------------------------------------------------------------

def parse_client_jsonl(path: Path):
    """Return list of latency_ms values from a client JSONL file."""
    latencies = []
    if not path.exists():
        return latencies
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                latencies.append(int(obj["latency_ms"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return latencies


def summarise_client(latencies):
    if not latencies:
        return {"n": 0, "p50": None, "p95": None, "p99": None}
    return {
        "n":   len(latencies),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
    }


# ---------------------------------------------------------------------------
# Directory layout understanding
# ---------------------------------------------------------------------------
# dvfs/runs/
#   llama_phase_matrix_YYYYMMDD_HHMMSS/
#     caseD_lc_be_long_policy/
#       dvfs_log.csv
#
# llama_validation/runs/
#   llama_phase_repeated_all_YYYYMMDD_HHMMSS/
#     run_001/
#       caseD_lc_be_long_policy/
#         lc_client.jsonl
#         be_client.jsonl
#         config.txt

CASE_ORDER = [
    "caseA_lc_alone_none",
    "caseB_be_long_alone_none",
    "caseC_lc_be_long_none",
    "caseD_lc_be_long_policy",
    "caseE_lc_be_short_none",
    "caseF_lc_be_short_policy",
    "caseG_lc_first_be_long_none",
    "caseH_lc_first_be_long_policy",
    "caseI_lc_cont_alone_none",
    "caseJ_lc_cont_be_long_none",
    "caseK_lc_cont_be_long_policy",
]

def case_sort_key(name):
    for i, c in enumerate(CASE_ORDER):
        if name.startswith(c[:5]):
            return i
    return 99


def read_config(case_dir: Path):
    cfg = {}
    p = case_dir / "config.txt"
    if p.exists():
        with p.open() as f:
            for line in f:
                if "=" in line:
                    k, _, v = line.strip().partition("=")
                    cfg[k.strip()] = v.strip()
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    default_dvfs  = Path.home() / "FYP/dvfs/runs"
    default_llama = Path.home() / "FYP/llama_validation/runs"

    dvfs_root  = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dvfs
    llama_root = Path(sys.argv[2]) if len(sys.argv) > 2 else default_llama

    out_dir = dvfs_root.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"DVFS runs:   {dvfs_root}")
    print(f"Llama runs:  {llama_root}")
    print(f"Output:      {out_dir}")
    print()

    # ------------------------------------------------------------------
    # 1. Discover DVFS run directories
    # ------------------------------------------------------------------
    # dvfs_runs/ contains one dir per matrix run
    # (each named llama_phase_matrix_YYYYMMDD_HHMMSS)
    dvfs_run_dirs = sorted(dvfs_root.glob("llama_phase_matrix_*"))
    if not dvfs_run_dirs:
        print(f"ERROR: no llama_phase_matrix_* directories found under {dvfs_root}")
        print("Check DVFS_RUN_ROOT and that at least one run completed.")
        sys.exit(1)

    print(f"Found {len(dvfs_run_dirs)} DVFS run directories.")

    # ------------------------------------------------------------------
    # 2. Parse DVFS logs
    # ------------------------------------------------------------------
    dvfs_rows = []   # one dict per (run, case)

    for run_dir in dvfs_run_dirs:
        case_dirs = sorted(
            [d for d in run_dir.iterdir() if d.is_dir()],
            key=lambda d: case_sort_key(d.name)
        )
        for case_dir in case_dirs:
            log = case_dir / "dvfs_log.csv"
            rows = parse_dvfs_log(log)
            cfg  = read_config(case_dir)
            m    = summarise_dvfs(rows)

            dvfs_rows.append({
                "run_dir":                run_dir.name,
                "case_name":              case_dir.name,
                "policy":                 cfg.get("policy", ""),
                **{k: fmt(v, 3) if isinstance(v, float) else ("" if v is None else v)
                   for k, v in m.items()},
            })

    # Write dvfs_summary.csv
    dvfs_out = out_dir / "dvfs_summary.csv"
    dvfs_fields = [
        "run_dir", "case_name", "policy", "n_samples",
        "cap_binding_frac",
        "mean_smClock_mhz", "lc_mean_clock_mhz",
        "be_only_mean_clock_mhz", "idle_mean_clock_mhz",
        "n_actuations",
        "lc_mean_limit_mw", "be_only_mean_limit_mw", "idle_mean_limit_mw",
        "mean_temp_c",
    ]
    with dvfs_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dvfs_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(dvfs_rows)

    print(f"Written: {dvfs_out}  ({len(dvfs_rows)} rows)")

    # ------------------------------------------------------------------
    # 3. Parse llama latency results
    # ------------------------------------------------------------------
    # Find the repeated run master directory
    llama_rows = []
    master_dirs = sorted(llama_root.glob("llama_phase_repeated_*"))

    if not master_dirs:
        # Fall back: look for individual matrix runs directly
        master_dirs = sorted(llama_root.glob("llama_phase_matrix_*"))

    if not master_dirs:
        print(f"WARNING: no llama run directories found under {llama_root}")
        print("         Skipping latency output.")
    else:
        for master_dir in master_dirs:
            # Under a repeated master, runs are run_001/ run_002/ etc.
            # Under a plain matrix run, cases are direct children.
            run_subdirs = sorted(d for d in master_dir.glob("run_*") if d.is_dir())
            if run_subdirs:
                # repeated run structure
                for run_sub in run_subdirs:
                    case_dirs = sorted(
                        [d for d in run_sub.iterdir() if d.is_dir()],
                        key=lambda d: case_sort_key(d.name)
                    )
                    for case_dir in case_dirs:
                        lc_lat = summarise_client(
                            parse_client_jsonl(case_dir / "lc_client.jsonl"))
                        be_lat = summarise_client(
                            parse_client_jsonl(case_dir / "be_client.jsonl"))
                        cfg = read_config(case_dir)
                        llama_rows.append({
                            "master_dir":       master_dir.name,
                            "run_dir":          run_sub.name,
                            "case_name":        case_dir.name,
                            "policy":           cfg.get("policy", ""),
                            "dvfs_enable":      cfg.get("dvfs_enable", ""),
                            "lc_client_p50_ms": fmt(lc_lat["p50"], 1),
                            "lc_client_p95_ms": fmt(lc_lat["p95"], 1),
                            "lc_client_p99_ms": fmt(lc_lat["p99"], 1),
                            "be_client_p50_ms": fmt(be_lat["p50"], 1),
                            "be_client_p95_ms": fmt(be_lat["p95"], 1),
                            "be_client_p99_ms": fmt(be_lat["p99"], 1),
                            "n_lc_requests":    lc_lat["n"],
                            "n_be_requests":    be_lat["n"],
                        })
            else:
                # plain matrix: cases are direct children
                case_dirs = sorted(
                    [d for d in master_dir.iterdir() if d.is_dir()],
                    key=lambda d: case_sort_key(d.name)
                )
                for case_dir in case_dirs:
                    lc_lat = summarise_client(
                        parse_client_jsonl(case_dir / "lc_client.jsonl"))
                    be_lat = summarise_client(
                        parse_client_jsonl(case_dir / "be_client.jsonl"))
                    cfg = read_config(case_dir)
                    llama_rows.append({
                        "master_dir":       master_dir.name,
                        "run_dir":          "",
                        "case_name":        case_dir.name,
                        "policy":           cfg.get("policy", ""),
                            "dvfs_enable":      cfg.get("dvfs_enable", ""),
                        "lc_client_p50_ms": fmt(lc_lat["p50"], 1),
                        "lc_client_p95_ms": fmt(lc_lat["p95"], 1),
                        "lc_client_p99_ms": fmt(lc_lat["p99"], 1),
                        "be_client_p50_ms": fmt(be_lat["p50"], 1),
                        "be_client_p95_ms": fmt(be_lat["p95"], 1),
                        "be_client_p99_ms": fmt(be_lat["p99"], 1),
                        "n_lc_requests":    lc_lat["n"],
                        "n_be_requests":    be_lat["n"],
                    })

    if llama_rows:
        llama_out = out_dir / "llama_dvfs_summary.csv"
        llama_fields = [
            "master_dir", "run_dir", "case_name", "policy", "dvfs_enable",
            "lc_client_p50_ms", "lc_client_p95_ms", "lc_client_p99_ms",
            "be_client_p50_ms", "be_client_p95_ms", "be_client_p99_ms",
            "n_lc_requests", "n_be_requests",
        ]
        with llama_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=llama_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(llama_rows)
        print(f"Written: {llama_out}  ({len(llama_rows)} rows)")

    # ------------------------------------------------------------------
    # 4. Quick terminal summary — the numbers you actually need right now
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("QUICK SUMMARY — averaged over all runs, key cases only")
    print("=" * 70)

    # Group dvfs_rows by case
    from collections import defaultdict
    dvfs_by_case = defaultdict(list)
    for row in dvfs_rows:
        dvfs_by_case[row["case_name"]].append(row)

    KEY_CASES = [
        ("caseA_lc_alone_none",        "LC alone (no policy)"),
        ("caseC_lc_be_long_none",      "LC + BE-long (no policy)"),
        ("caseD_lc_be_long_policy",    "LC + BE-long (CAP policy)"),
        ("caseI_lc_cont_alone_none",   "Cont. LC alone (no policy)"),
        ("caseJ_lc_cont_be_long_none", "Cont. LC + BE-long (no policy)"),
        ("caseK_lc_cont_be_long_policy","Cont. LC + BE-long (CAP policy)"),
    ]

    def avg(rows, field):
        vals = []
        for r in rows:
            v = r.get(field, "")
            if v != "":
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        return safe_mean(vals)

    print(f"\n{'Case':<42} {'CapBind%':>8} {'Clock(all)':>10} "
          f"{'Clock(LC)':>10} {'Clock(BE)':>10} {'Actuations':>10}")
    print("-" * 96)

    for case_key, label in KEY_CASES:
        # Match on prefix since the directory name might differ slightly
        matching = [r for name, rows in dvfs_by_case.items()
                    for r in rows if name.startswith(case_key[:10])]
        if not matching:
            print(f"  {label:<40}  (no data)")
            continue

        cb    = avg(matching, "cap_binding_frac")
        clk   = avg(matching, "mean_smClock_mhz")
        lclk  = avg(matching, "lc_mean_clock_mhz")
        bclk  = avg(matching, "be_only_mean_clock_mhz")
        acts  = avg(matching, "n_actuations")

        print(f"  {label:<40}  "
              f"{fmt(cb*100 if cb else None,1):>7}%  "
              f"{fmt(clk,0):>10}  "
              f"{fmt(lclk,0):>10}  "
              f"{fmt(bclk,0):>10}  "
              f"{fmt(acts,1):>10}")

    print()
    print("cap_binding_frac: fraction of samples where power draw > 85% of limit.")
    print("If this is low, the power cap was not constraining Boost under llama.cpp")
    print("(consistent with memory-bound decode being the dominant phase).")
    print()
    print("Full data: dvfs_summary.csv and llama_dvfs_summary.csv in dvfs/analysis/")


if __name__ == "__main__":
    main()
