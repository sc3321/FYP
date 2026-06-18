#!/usr/bin/env python3
"""
Analyse the overhead experiment output.

Usage:
    python3 analyse_overhead.py runs/overhead_<timestamp>/
    python3 analyse_overhead.py runs/overhead_<timestamp>/ --n-bootstrap 10000

For each replicate, reads vanilla/lc_client.jsonl and instr/lc_client.jsonl.
Reports pooled percentiles with bootstrap CIs and a per-replicate sign test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


# ---------- IO ----------------------------------------------------------------

def read_latencies(jsonl_path: Path) -> np.ndarray:
    """Return latency_ms values from a client jsonl file. Empty array if missing."""
    if not jsonl_path.exists():
        return np.array([], dtype=float)
    lats: list[float] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "latency_ms" in rec:
                lats.append(float(rec["latency_ms"]))
    return np.array(lats, dtype=float)


def discover_replicates(root: Path) -> list[Path]:
    reps = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("rep")])
    if not reps:
        sys.exit(f"ERROR: no rep* directories under {root}")
    return reps


# ---------- Stats -------------------------------------------------------------

def percentile(arr: np.ndarray, p: float) -> float:
    """Numpy linear-interpolation percentile. Returns nan on empty."""
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def bootstrap_ci(
    arr: np.ndarray,
    p: float,
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile-bootstrap CI for the p-th percentile of `arr`."""
    if arr.size == 0:
        return (float("nan"), float("nan"))
    n = arr.size
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    samples = arr[idx]
    stats = np.percentile(samples, p, axis=1)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def sign_test_pvalue(k: int, n: int) -> float:
    """Two-sided exact binomial p-value for k successes out of n under p=0.5."""
    from math import comb
    if n == 0:
        return float("nan")
    # Two-sided: sum of point masses at least as extreme as k.
    p_one_sided = sum(comb(n, j) for j in range(k, n + 1)) / (2 ** n)
    # Reflect for two-sided.
    p_two_sided = min(1.0, 2 * min(p_one_sided, 1 - p_one_sided + comb(n, k) / (2 ** n)))
    # Simpler: probability of |X - n/2| >= |k - n/2|.
    target = abs(k - n / 2)
    total = sum(
        comb(n, j) for j in range(n + 1) if abs(j - n / 2) >= target
    ) / (2 ** n)
    return float(total)


# ---------- Per-replicate summary --------------------------------------------

def per_replicate_stats(reps: Sequence[Path]) -> dict:
    """Return per-replicate p50/p95/p99 for vanilla and instr arms."""
    out = {
        "rep": [],
        "n_vanilla": [], "n_instr": [],
        "vanilla_p50": [], "vanilla_p95": [], "vanilla_p99": [],
        "instr_p50":   [], "instr_p95":   [], "instr_p99":   [],
    }
    for rep_dir in reps:
        v = read_latencies(rep_dir / "vanilla" / "lc_client.jsonl")
        i = read_latencies(rep_dir / "instr"   / "lc_client.jsonl")
        out["rep"].append(rep_dir.name)
        out["n_vanilla"].append(int(v.size))
        out["n_instr"].append(int(i.size))
        out["vanilla_p50"].append(percentile(v, 50))
        out["vanilla_p95"].append(percentile(v, 95))
        out["vanilla_p99"].append(percentile(v, 99))
        out["instr_p50"].append(percentile(i, 50))
        out["instr_p95"].append(percentile(i, 95))
        out["instr_p99"].append(percentile(i, 99))
    return out


# ---------- Pooled summary ----------------------------------------------------

def pooled_summary(
    reps: Sequence[Path],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    vanilla_all = np.concatenate([
        read_latencies(r / "vanilla" / "lc_client.jsonl") for r in reps
    ])
    instr_all = np.concatenate([
        read_latencies(r / "instr" / "lc_client.jsonl") for r in reps
    ])

    res = {"vanilla_n": int(vanilla_all.size), "instr_n": int(instr_all.size)}
    for p in (50, 95, 99):
        v = percentile(vanilla_all, p)
        i = percentile(instr_all, p)
        v_lo, v_hi = bootstrap_ci(vanilla_all, p, n_bootstrap, rng)
        i_lo, i_hi = bootstrap_ci(instr_all,   p, n_bootstrap, rng)
        delta_pct = ((i - v) / v * 100.0) if v else float("nan")
        res[f"vanilla_p{p}"]    = v
        res[f"vanilla_p{p}_lo"] = v_lo
        res[f"vanilla_p{p}_hi"] = v_hi
        res[f"instr_p{p}"]      = i
        res[f"instr_p{p}_lo"]   = i_lo
        res[f"instr_p{p}_hi"]   = i_hi
        res[f"delta_p{p}_pct"]  = delta_pct
    return res


# ---------- Directional per-replicate comparison ------------------------------

def directional_comparison(per_rep: dict, metric: str) -> dict:
    """For each replicate, compare vanilla vs instr at one percentile."""
    v = np.array(per_rep[f"vanilla_{metric}"])
    i = np.array(per_rep[f"instr_{metric}"])
    valid = ~(np.isnan(v) | np.isnan(i))
    v = v[valid]; i = i[valid]
    deltas_abs = i - v                    # ms
    deltas_pct = (i - v) / v * 100.0      # %
    instr_higher = int(np.sum(i > v))
    ties = int(np.sum(i == v))
    n = int(v.size)
    p = sign_test_pvalue(instr_higher, n - ties) if (n - ties) > 0 else float("nan")
    return {
        "n": n,
        "ties": ties,
        "instr_higher_than_vanilla": instr_higher,
        "instr_lower_than_vanilla": n - ties - instr_higher,
        "mean_delta_ms": float(np.mean(deltas_abs)),
        "std_delta_ms":  float(np.std(deltas_abs, ddof=1)) if n > 1 else float("nan"),
        "mean_delta_pct": float(np.mean(deltas_pct)),
        "std_delta_pct":  float(np.std(deltas_pct, ddof=1)) if n > 1 else float("nan"),
        "sign_test_p_two_sided": p,
    }


# ---------- Output ------------------------------------------------------------

def fmt_ms(x: float) -> str:
    if np.isnan(x): return "   nan"
    return f"{x:7.1f}"

def fmt_pct(x: float) -> str:
    if np.isnan(x): return "  nan%"
    return f"{x:+6.2f}%"

def print_pooled(p: dict) -> None:
    print()
    print("=" * 72)
    print("POOLED ACROSS ALL REPLICATES (matches §7.2.4 summary method)")
    print("=" * 72)
    print(f"  vanilla n = {p['vanilla_n']}  events")
    print(f"  instr   n = {p['instr_n']}  events")
    print()
    print(f"  {'metric':<8}{'vanilla (ms)':>22}{'instr (ms)':>22}{'delta':>14}")
    for q in (50, 95, 99):
        v_str = f"{p[f'vanilla_p{q}']:.1f} [{p[f'vanilla_p{q}_lo']:.1f}, {p[f'vanilla_p{q}_hi']:.1f}]"
        i_str = f"{p[f'instr_p{q}']:.1f} [{p[f'instr_p{q}_lo']:.1f}, {p[f'instr_p{q}_hi']:.1f}]"
        d_str = fmt_pct(p[f'delta_p{q}_pct'])
        print(f"  p{q:<7}{v_str:>22}{i_str:>22}{d_str:>14}")
    print()
    print("  Brackets are 95% bootstrap CIs (10,000 resamples by default).")
    print("  Non-overlapping CIs are a sufficient (but not necessary) condition")
    print("  for the difference to be considered robust.")


def print_directional(per_rep: dict) -> None:
    print()
    print("=" * 72)
    print("PER-REPLICATE PAIRED DIRECTIONAL COMPARISON")
    print("=" * 72)
    for metric, label in [("p50", "p50"), ("p95", "p95"), ("p99", "p99")]:
        d = directional_comparison(per_rep, metric)
        print()
        print(f"  metric: {label}")
        print(f"    n replicates              = {d['n']}  (ties: {d['ties']})")
        print(f"    instr  higher  than vanilla = {d['instr_higher_than_vanilla']} / {d['n'] - d['ties']}")
        print(f"    instr  lower   than vanilla = {d['instr_lower_than_vanilla']} / {d['n'] - d['ties']}")
        print(f"    mean delta (ms)           = {d['mean_delta_ms']:+.2f}")
        print(f"    std  delta (ms)           = {d['std_delta_ms']:.2f}")
        print(f"    mean delta (%)            = {d['mean_delta_pct']:+.2f}")
        print(f"    std  delta (%)            = {d['std_delta_pct']:.2f}")
        print(f"    sign-test p (two-sided)   = {d['sign_test_p_two_sided']:.4g}")


def write_per_replicate_csv(per_rep: dict, out_path: Path) -> None:
    import csv
    cols = list(per_rep.keys())
    n = len(per_rep["rep"])
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in range(n):
            w.writerow([per_rep[c][k] for c in cols])
    print(f"\nWrote per-replicate CSV: {out_path}")


def write_pooled_csv(pooled: dict, directional: dict, out_path: Path) -> None:
    import csv
    rows = []
    for q in (50, 95, 99):
        rows.append({
            "metric": f"p{q}",
            "vanilla_ms": pooled[f"vanilla_p{q}"],
            "vanilla_ci_lo": pooled[f"vanilla_p{q}_lo"],
            "vanilla_ci_hi": pooled[f"vanilla_p{q}_hi"],
            "instr_ms": pooled[f"instr_p{q}"],
            "instr_ci_lo": pooled[f"instr_p{q}_lo"],
            "instr_ci_hi": pooled[f"instr_p{q}_hi"],
            "pooled_delta_pct": pooled[f"delta_p{q}_pct"],
            "per_rep_n": directional[f"p{q}"]["n"],
            "per_rep_instr_higher": directional[f"p{q}"]["instr_higher_than_vanilla"],
            "per_rep_instr_lower":  directional[f"p{q}"]["instr_lower_than_vanilla"],
            "per_rep_ties": directional[f"p{q}"]["ties"],
            "per_rep_mean_delta_pct": directional[f"p{q}"]["mean_delta_pct"],
            "per_rep_std_delta_pct":  directional[f"p{q}"]["std_delta_pct"],
            "per_rep_sign_p_two_sided": directional[f"p{q}"]["sign_test_p_two_sided"],
        })
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote summary CSV: {out_path}")


# ---------- main --------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="overhead_<timestamp> directory")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.run_dir.is_dir():
        sys.exit(f"ERROR: not a directory: {args.run_dir}")

    rng = np.random.default_rng(args.seed)
    reps = discover_replicates(args.run_dir)

    print(f"Run directory: {args.run_dir}")
    print(f"Replicates discovered: {len(reps)}")

    per_rep = per_replicate_stats(reps)
    pooled = pooled_summary(reps, args.n_bootstrap, rng)

    print_pooled(pooled)
    print_directional(per_rep)

    directional = {
        metric: directional_comparison(per_rep, metric)
        for metric in ("p50", "p95", "p99")
    }

    write_per_replicate_csv(per_rep, args.run_dir / "overhead_per_replicate.csv")
    write_pooled_csv(pooled, directional, args.run_dir / "overhead_summary.csv")

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
