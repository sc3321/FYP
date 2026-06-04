#!/usr/bin/env python3
"""
Pooled cross-run aggregator for the llama phase matrix.

Walks a repeated-run master dir produced by run_x_llama_phase_matrices.sh:

  master_dir/
    run_001/
      caseA_lc_alone_none/
        intervals.csv          <-- phase-instrumented (LLAMA_REQUEST, LLAMA_DECODE)
        lc_client.jsonl        <-- curl-wrapped client latencies
        be_client.jsonl
        summary.json
        ...
      caseB_be_long_alone_none/
      ...
    run_002/
    ...

For each case, it pools raw per-request observations across all reps and
computes p50/p95/p99 with bootstrap 95% CIs.

Why this is different from your existing aggregator:
  Your existing aggregator (aggregate_llama_runs.py) loads each rep's
  summary.csv, which already contains that rep's p99 over ~50 requests, and
  then takes the median of those 30 p99 values. That smooths the tail away
  -- p99 from 50 obs is noisy, and median-across-reps is biased low for
  tail quantiles. This script pools all 30 reps' raw observations into one
  ~1500-point sample and computes one p99 directly, with a bootstrap CI.

  For comparison, it also reports per-rep-median-p99 (what the old script
  gives you) alongside pooled-p99 so you can see the gap.

Metrics produced per case:
  lc_client_*        : curl-wrapped LC client latency
  lc_request_*       : LLAMA_REQUEST phase duration, class=LC
  lc_decode_*        : LLAMA_DECODE phase duration, class=LC
  be_client_*        : curl-wrapped BE client latency
  be_request_*       : LLAMA_REQUEST phase duration, class=BE (all)
  be_long_request_*  : LLAMA_REQUEST phase duration, class=BE granularity=LONG
  be_short_request_* : LLAMA_REQUEST phase duration, class=BE granularity=SHORT
  be_decode_*        : LLAMA_DECODE phase duration, class=BE (all)
  be_long_decode_*   : LLAMA_DECODE phase duration, class=BE granularity=LONG
  be_short_decode_*  : LLAMA_DECODE phase duration, class=BE granularity=SHORT
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable
import numpy as np


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------

def load_client_jsonl(path: Path) -> list[float]:
    """Load latency_ms values from a {lc,be}_client.jsonl file."""
    out: list[float] = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = rec.get("latency_ms")
            if v is not None:
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    pass
    return out


def load_intervals(path: Path) -> list[dict]:
    """Load rows from a case's intervals.csv (written by analyse_llama_phase_matrix.py)."""
    out: list[dict] = []
    if not path.exists():
        return out
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out.append({
                    "dur_ms": float(row["dur_ms"]),
                    "phase_type": (row.get("phase_type") or "").strip().upper(),
                    "class": (row.get("class") or "").strip().upper(),
                    "granularity": (row.get("granularity") or "").strip().upper(),
                })
            except (KeyError, ValueError):
                continue
    return out


def filter_durs(
    intervals: Iterable[dict],
    phase_type: str | None,
    cls: str | None,
    granularity: str | None,
) -> list[float]:
    out: list[float] = []
    for row in intervals:
        if phase_type is not None and row["phase_type"] != phase_type:
            continue
        if cls is not None and row["class"] != cls:
            continue
        if granularity is not None and row["granularity"] != granularity:
            continue
        out.append(row["dur_ms"])
    return out


# ----------------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------------

def bootstrap_ci(
    data: np.ndarray,
    q: float,
    n_boot: int = 2000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for a quantile estimate.

    Resample the data with replacement n_boot times, recompute the quantile
    each time, then take the alpha/2 and 1-alpha/2 quantiles of the bootstrap
    distribution. For n ~ 1500 and q=99, supported by ~15 datapoints in each
    resample -- enough to be stable, not enough to be precise. Report the CI.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)
    n = len(data)
    if n < 2:
        return (float("nan"), float("nan"))
    # Vectorised resample: (n_boot, n) matrix of indices, then percentile per row.
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = data[idx]
    boots = np.percentile(samples, q, axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.percentile(boots, 100 * alpha)), float(np.percentile(boots, 100 * (1 - alpha)))


def pooled_stats(values: list[float], per_rep_values: list[list[float]]) -> dict:
    """
    Compute pooled percentile stats and the per-rep-median equivalent for comparison.

    Returns NaN where data is insufficient.
    """
    if not values:
        return {
            "n_obs": 0,
            "n_reps_with_data": 0,
            "pooled_p50": float("nan"),
            "pooled_p95": float("nan"),
            "pooled_p95_ci_lo": float("nan"),
            "pooled_p95_ci_hi": float("nan"),
            "pooled_p99": float("nan"),
            "pooled_p99_ci_lo": float("nan"),
            "pooled_p99_ci_hi": float("nan"),
            "per_rep_p99_median": float("nan"),
            "per_rep_p99_min": float("nan"),
            "per_rep_p99_max": float("nan"),
        }

    arr = np.array(values, dtype=float)

    per_rep_p99s = [float(np.percentile(v, 99)) for v in per_rep_values if v]

    p95_lo, p95_hi = bootstrap_ci(arr, 95)
    p99_lo, p99_hi = bootstrap_ci(arr, 99)

    return {
        "n_obs": int(arr.size),
        "n_reps_with_data": len(per_rep_p99s),
        "pooled_p50": float(np.percentile(arr, 50)),
        "pooled_p95": float(np.percentile(arr, 95)),
        "pooled_p95_ci_lo": p95_lo,
        "pooled_p95_ci_hi": p95_hi,
        "pooled_p99": float(np.percentile(arr, 99)),
        "pooled_p99_ci_lo": p99_lo,
        "pooled_p99_ci_hi": p99_hi,
        "per_rep_p99_median": float(np.median(per_rep_p99s)) if per_rep_p99s else float("nan"),
        "per_rep_p99_min": float(min(per_rep_p99s)) if per_rep_p99s else float("nan"),
        "per_rep_p99_max": float(max(per_rep_p99s)) if per_rep_p99s else float("nan"),
    }


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

# (metric_name, source, phase_type, class, granularity)
#   source = "lc_client" | "be_client" | "intervals"
METRICS: list[tuple[str, str, str | None, str | None, str | None]] = [
    ("lc_client",        "lc_client",  None,             None, None),
    ("be_client",        "be_client",  None,             None, None),

    ("lc_request",       "intervals",  "LLAMA_REQUEST",  "LC", None),
    ("lc_decode",        "intervals",  "LLAMA_DECODE",   "LC", None),

    ("be_request",       "intervals",  "LLAMA_REQUEST",  "BE", None),
    ("be_long_request",  "intervals",  "LLAMA_REQUEST",  "BE", "LONG"),
    ("be_short_request", "intervals",  "LLAMA_REQUEST",  "BE", "SHORT"),

    ("be_decode",        "intervals",  "LLAMA_DECODE",   "BE", None),
    ("be_long_decode",   "intervals",  "LLAMA_DECODE",   "BE", "LONG"),
    ("be_short_decode",  "intervals",  "LLAMA_DECODE",   "BE", "SHORT"),
]


def aggregate_case(case_name: str, rep_dirs: list[Path]) -> dict:
    """For one case, pool across all reps and compute stats per metric."""
    # Collect raw observations per metric, plus per-rep lists for the
    # per-rep-p99 comparison statistic.
    pooled: dict[str, list[float]] = {m[0]: [] for m in METRICS}
    per_rep: dict[str, list[list[float]]] = {m[0]: [] for m in METRICS}

    for rep_dir in rep_dirs:
        case_dir = rep_dir / case_name
        if not case_dir.is_dir():
            continue

        lc_client = load_client_jsonl(case_dir / "lc_client.jsonl")
        be_client = load_client_jsonl(case_dir / "be_client.jsonl")
        intervals = load_intervals(case_dir / "intervals.csv")

        for name, source, ptype, cls, gran in METRICS:
            if source == "lc_client":
                vals = lc_client
            elif source == "be_client":
                vals = be_client
            elif source == "intervals":
                vals = filter_durs(intervals, ptype, cls, gran)
            else:
                vals = []

            pooled[name].extend(vals)
            if vals:
                per_rep[name].append(vals)

    out: dict = {"case": case_name}
    for name, *_ in METRICS:
        stats = pooled_stats(pooled[name], per_rep[name])
        for k, v in stats.items():
            out[f"{name}_{k}"] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("master_dir", type=Path,
                   help="Repeated-run master directory containing run_001/, run_002/, ...")
    p.add_argument("--rep-glob", default="run_[0-9][0-9][0-9]",
                   help="Glob for rep directories. Default matches run_NNN.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV path. Default: <master_dir>/aggregate_pooled.csv")
    args = p.parse_args()

    if not args.master_dir.is_dir():
        raise SystemExit(f"not a directory: {args.master_dir}")

    rep_dirs = sorted(args.master_dir.glob(args.rep_glob))
    rep_dirs = [d for d in rep_dirs if d.is_dir()]
    if not rep_dirs:
        raise SystemExit(f"no rep dirs matched {args.rep_glob!r} under {args.master_dir}")

    # Discover case names from the first rep dir.
    case_names = sorted(d.name for d in rep_dirs[0].iterdir() if d.is_dir())
    if not case_names:
        raise SystemExit(f"no case dirs under {rep_dirs[0]}")

    print(f"Master dir:  {args.master_dir}")
    print(f"Rep dirs:    {len(rep_dirs)}")
    print(f"Cases:       {len(case_names)}")
    print()

    rows = [aggregate_case(name, rep_dirs) for name in case_names]

    # Stable column ordering: case, then per-metric blocks of 11 columns.
    metric_keys = [
        "n_obs", "n_reps_with_data",
        "pooled_p50", "pooled_p95", "pooled_p95_ci_lo", "pooled_p95_ci_hi",
        "pooled_p99", "pooled_p99_ci_lo", "pooled_p99_ci_hi",
        "per_rep_p99_median", "per_rep_p99_min", "per_rep_p99_max",
    ]
    fieldnames = ["case"]
    for name, *_ in METRICS:
        fieldnames.extend(f"{name}_{k}" for k in metric_keys)

    out_path = args.out or (args.master_dir / "aggregate_pooled.csv")

    def fmt(v: object) -> str:
        if isinstance(v, float):
            if np.isnan(v):
                return ""
            return f"{v:.3f}"
        return str(v) if v is not None else ""

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: fmt(row.get(k, "")) for k in fieldnames})

    print(f"Wrote: {out_path}")
    print()

    # Print a compact comparison for the headline metrics, side-by-side.
    print("Headline comparison (LC p99, ms):")
    print(f"{'case':40s}  {'n':>5s}  {'pooled':>9s}  {'95% CI':>20s}  {'per-rep-med':>11s}")
    for row in rows:
        n = row["lc_request_n_obs"]
        pp = row["lc_request_pooled_p99"]
        lo = row["lc_request_pooled_p99_ci_lo"]
        hi = row["lc_request_pooled_p99_ci_hi"]
        med = row["lc_request_per_rep_p99_median"]
        if isinstance(pp, float) and np.isnan(pp):
            print(f"{row['case']:40s}  {'-':>5s}  {'-':>9s}  {'-':>20s}  {'-':>11s}")
            continue
        ci = f"[{lo:>7.1f}, {hi:>7.1f}]"
        print(f"{row['case']:40s}  {n:5d}  {pp:9.1f}  {ci:>20s}  {med:11.1f}")


if __name__ == "__main__":
    main()
