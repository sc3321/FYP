#!/usr/bin/env python3
"""
plot_phase3_results.py

Generate Phase 3 evaluation plots from grouped_tradeoff.csv files.

Accepts either:
  - one or more grouped_tradeoff.csv files
  - one or more output directories containing analysis/grouped_tradeoff.csv
  - one or more parent directories; the script searches recursively for grouped_tradeoff.csv

Main outputs:
  - worker_scaling_lc_p99.{png,pdf}
  - worker_scaling_lc_p95.{png,pdf}
  - worker_scaling_be_throughput.{png,pdf}
  - delay_tradeoff_<case>_<worker>.{png,pdf}
  - delay_lc_tail_<case>_<worker>.{png,pdf}
  - delay_be_throughput_<case>_<worker>.{png,pdf}
  - delay_overlap_<case>_<worker>.{png,pdf}
  - delay_overlap_count_<case>_<worker>.{png,pdf}
  - combined_long_vs_mixed_lc_p99.{png,pdf}
  - combined_long_vs_mixed_overlap.{png,pdf}
  - plot_data_combined.csv

Expected grouped_tradeoff.csv columns:
  case, worker_config, policy, delay_us, total_be_workers,
  long_be_workers, chunked_be_workers, start_order, n,
  lc_p50_ms_mean, lc_p95_ms_mean, lc_p99_ms_mean, lc_max_ms_mean,
  be_agg_thr_s_mean, be_per_proc_thr_s_mean, be_wall_ms_mean,
  be_total_active_ms_mean, be_phase_count_mean,
  overlap_union_ms_mean, overlap_weighted_ms_mean,
  overlap_count_mean, max_concurrent_be_mean
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


NUMERIC_COLUMNS = [
    "delay_us",
    "total_be_workers",
    "long_be_workers",
    "chunked_be_workers",
    "n",
    "lc_p50_ms_mean",
    "lc_p95_ms_mean",
    "lc_p99_ms_mean",
    "lc_max_ms_mean",
    "be_agg_thr_s_mean",
    "be_per_proc_thr_s_mean",
    "be_wall_ms_mean",
    "be_total_active_ms_mean",
    "be_phase_count_mean",
    "overlap_union_ms_mean",
    "overlap_weighted_ms_mean",
    "overlap_count_mean",
    "max_concurrent_be_mean",
]


def find_tradeoff_files(inputs: Iterable[str]) -> List[Path]:
    files: List[Path] = []

    for raw in inputs:
        p = Path(raw)

        if p.is_file():
            files.append(p)
            continue

        if p.is_dir():
            direct = p / "analysis" / "grouped_tradeoff.csv"
            if direct.exists():
                files.append(direct)
                continue

            files.extend(sorted(p.rglob("grouped_tradeoff.csv")))
            continue

        print(f"[WARN] input does not exist: {p}")

    # de-duplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for f in files:
        resolved = f.resolve()
        if resolved not in seen:
            unique.append(f)
            seen.add(resolved)

    return unique


def read_tradeoff_files(files: List[Path]) -> pd.DataFrame:
    frames = []

    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = str(f)
        frames.append(df)

    if not frames:
        raise SystemExit("No grouped_tradeoff.csv files found.")

    df = pd.concat(frames, ignore_index=True)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["case", "worker_config", "policy", "start_order"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    # Remove exact duplicate rows from accidental repeated inputs.
    subset_cols = [
        c for c in [
            "case",
            "worker_config",
            "policy",
            "delay_us",
            "total_be_workers",
            "long_be_workers",
            "chunked_be_workers",
            "start_order",
            "lc_p95_ms_mean",
            "lc_p99_ms_mean",
            "be_agg_thr_s_mean",
            "overlap_weighted_ms_mean",
        ]
        if c in df.columns
    ]
    df = df.drop_duplicates(subset=subset_cols)

    return df


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def savefig(out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"

    plt.tight_layout()
    plt.savefig(png, dpi=220)
    plt.savefig(pdf)
    plt.close()

    print(f"[PLOT] wrote {png}")
    print(f"[PLOT] wrote {pdf}")


def annotate_points(ax, xs, ys, labels: Optional[List[str]] = None) -> None:
    if labels is None:
        labels = [str(x) for x in xs]

    for x, y, label in zip(xs, ys, labels):
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )


def filter_worker_scaling_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pick rows suitable for worker-scaling plots.

    Preference:
      1. policy == none
      2. else delay_us == 0
      3. else smallest delay per case/worker/policy
    """
    if "policy" in df.columns and (df["policy"] == "none").any():
        out = df[df["policy"] == "none"].copy()
    elif "delay_us" in df.columns and (df["delay_us"] == 0).any():
        out = df[df["delay_us"] == 0].copy()
    else:
        group_cols = ["case", "worker_config", "policy"]
        idx = df.groupby(group_cols)["delay_us"].idxmin()
        out = df.loc[idx].copy()

    # Only colocated LC-vs-BE rows are useful for worker scaling.
    out = out[out["case"].str.startswith("lc_vs")].copy()

    worker_order = {
        "long1": 1,
        "long4": 2,
        "chunked1": 3,
        "chunked4": 4,
        "mixed2": 5,
        "mixed4": 6,
    }
    out["worker_order"] = out["worker_config"].map(worker_order).fillna(99)
    out = out.sort_values(["case", "worker_order", "worker_config"])

    return out


def plot_worker_scaling_metric(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str,
    ylabel: str,
    filename: str,
    title: str,
) -> None:
    rows = filter_worker_scaling_rows(df)

    if rows.empty or metric not in rows.columns:
        print(f"[SKIP] worker scaling {metric}: no rows")
        return

    labels = [
        f"{r.worker_config}\n{r.case.replace('lc_vs_be_', '').replace('lc_vs_', '')}"
        for r in rows.itertuples()
    ]

    plt.figure(figsize=(max(8, len(rows) * 0.8), 4.8))
    plt.bar(range(len(rows)), rows[metric])
    plt.xticks(range(len(rows)), labels, rotation=0)
    plt.ylabel(ylabel)
    plt.xlabel("BE worker configuration")
    plt.title(title)

    for i, value in enumerate(rows[metric]):
        plt.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    savefig(out_dir, filename)


def delay_groups(df: pd.DataFrame, policy: str = "proper") -> List[Tuple[str, str, pd.DataFrame]]:
    rows = df.copy()

    if "policy" in rows.columns:
        rows = rows[rows["policy"] == policy].copy()

    rows = rows[rows["case"].str.startswith("lc_vs")].copy()

    # Need at least two delay points to be a delay sweep.
    groups = []
    for (case, worker), g in rows.groupby(["case", "worker_config"]):
        g = g.sort_values("delay_us")
        if g["delay_us"].nunique() >= 2:
            groups.append((case, worker, g))

    return groups


def plot_delay_tradeoff(case: str, worker: str, g: pd.DataFrame, out_dir: Path) -> None:
    required = {"delay_us", "lc_p99_ms_mean", "be_agg_thr_s_mean"}
    if not required.issubset(g.columns):
        print(f"[SKIP] tradeoff {case}/{worker}: missing columns")
        return

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))

    ax1.plot(g["delay_us"], g["lc_p99_ms_mean"], marker="o", label="LC p99")
    ax1.plot(g["delay_us"], g["lc_p95_ms_mean"], marker="o", linestyle="--", label="LC p95")
    ax1.set_xlabel("BE delay at admission (µs)")
    ax1.set_ylabel("LC latency (ms)")
    ax1.set_xscale("log")
    ax1.grid(True, which="both", axis="both", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(g["delay_us"], g["be_agg_thr_s_mean"], marker="s", label="BE throughput")
    ax2.set_ylabel("BE throughput (phases/s)")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.title(f"LC tail latency vs BE throughput: {case}, {worker}")
    savefig(out_dir, f"delay_tradeoff_{safe_name(case)}_{safe_name(worker)}")


def plot_delay_lc_tail(case: str, worker: str, g: pd.DataFrame, out_dir: Path) -> None:
    if not {"delay_us", "lc_p95_ms_mean", "lc_p99_ms_mean"}.issubset(g.columns):
        return

    plt.figure(figsize=(7.2, 4.5))
    plt.plot(g["delay_us"], g["lc_p95_ms_mean"], marker="o", label="LC p95")
    plt.plot(g["delay_us"], g["lc_p99_ms_mean"], marker="o", label="LC p99")
    plt.xscale("log")
    plt.xlabel("BE delay at admission (µs)")
    plt.ylabel("LC latency (ms)")
    plt.title(f"LC tail latency under policy: {case}, {worker}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    savefig(out_dir, f"delay_lc_tail_{safe_name(case)}_{safe_name(worker)}")


def plot_delay_be_throughput(case: str, worker: str, g: pd.DataFrame, out_dir: Path) -> None:
    if not {"delay_us", "be_agg_thr_s_mean"}.issubset(g.columns):
        return

    plt.figure(figsize=(7.2, 4.5))
    plt.plot(g["delay_us"], g["be_agg_thr_s_mean"], marker="o")
    plt.xscale("log")
    plt.xlabel("BE delay at admission (µs)")
    plt.ylabel("BE throughput (phases/s)")
    plt.title(f"BE throughput under policy: {case}, {worker}")
    plt.grid(True, which="both", alpha=0.3)
    savefig(out_dir, f"delay_be_throughput_{safe_name(case)}_{safe_name(worker)}")


def plot_delay_overlap(case: str, worker: str, g: pd.DataFrame, out_dir: Path) -> None:
    if not {"delay_us", "overlap_weighted_ms_mean", "overlap_union_ms_mean"}.issubset(g.columns):
        return

    plt.figure(figsize=(7.2, 4.5))
    plt.plot(g["delay_us"], g["overlap_weighted_ms_mean"], marker="o", label="Weighted overlap")
    plt.plot(g["delay_us"], g["overlap_union_ms_mean"], marker="o", label="Union overlap")
    plt.xscale("log")
    plt.xlabel("BE delay at admission (µs)")
    plt.ylabel("LC/BE overlap (ms)")
    plt.title(f"LC/BE overlap under policy: {case}, {worker}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    savefig(out_dir, f"delay_overlap_{safe_name(case)}_{safe_name(worker)}")


def plot_delay_overlap_count(case: str, worker: str, g: pd.DataFrame, out_dir: Path) -> None:
    if not {"delay_us", "overlap_count_mean", "max_concurrent_be_mean"}.issubset(g.columns):
        return

    plt.figure(figsize=(7.2, 4.5))
    plt.plot(g["delay_us"], g["overlap_count_mean"], marker="o", label="Mean overlap count")
    plt.plot(g["delay_us"], g["max_concurrent_be_mean"], marker="o", label="Max concurrent BE")
    plt.xscale("log")
    plt.xlabel("BE delay at admission (µs)")
    plt.ylabel("Concurrent / overlapping BE phases")
    plt.title(f"BE concurrency during LC requests: {case}, {worker}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    savefig(out_dir, f"delay_overlap_count_{safe_name(case)}_{safe_name(worker)}")


def plot_combined_delay_metric(
    df: pd.DataFrame,
    out_dir: Path,
    workers: List[str],
    metric: str,
    ylabel: str,
    filename: str,
    title: str,
    policy: str = "proper",
) -> None:
    rows = df.copy()
    rows = rows[(rows["policy"] == policy) & (rows["worker_config"].isin(workers))]
    rows = rows[rows["case"].str.startswith("lc_vs")].copy()

    if rows.empty or metric not in rows.columns:
        print(f"[SKIP] combined {metric}: no rows")
        return

    plt.figure(figsize=(7.5, 4.8))

    plotted = False
    for (case, worker), g in rows.groupby(["case", "worker_config"]):
        g = g.sort_values("delay_us")
        if g["delay_us"].nunique() < 2:
            continue
        label = f"{case}, {worker}"
        plt.plot(g["delay_us"], g[metric], marker="o", label=label)
        plotted = True

    if not plotted:
        plt.close()
        print(f"[SKIP] combined {metric}: no multi-delay groups")
        return

    plt.xscale("log")
    plt.xlabel("BE delay at admission (µs)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    savefig(out_dir, filename)


def write_selected_summary_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_cols = [
        "case",
        "worker_config",
        "policy",
        "delay_us",
        "total_be_workers",
        "long_be_workers",
        "chunked_be_workers",
        "n",
        "lc_p95_ms_mean",
        "lc_p99_ms_mean",
        "be_agg_thr_s_mean",
        "be_wall_ms_mean",
        "overlap_union_ms_mean",
        "overlap_weighted_ms_mean",
        "overlap_count_mean",
        "max_concurrent_be_mean",
    ]
    selected_cols = [c for c in selected_cols if c in df.columns]

    df[selected_cols].sort_values(
        ["case", "worker_config", "policy", "delay_us"]
    ).to_csv(out_dir / "plot_data_combined.csv", index=False)

    # Worker scaling table.
    worker_rows = filter_worker_scaling_rows(df)
    if not worker_rows.empty:
        worker_rows[selected_cols].to_csv(out_dir / "table_worker_scaling.csv", index=False)

    # Delay sweep table.
    delay_rows = df[(df["policy"] == "proper") & (df["case"].str.startswith("lc_vs"))].copy()
    delay_rows = delay_rows[delay_rows.groupby(["case", "worker_config"])["delay_us"].transform("nunique") >= 2]
    if not delay_rows.empty:
        delay_rows[selected_cols].sort_values(
            ["case", "worker_config", "delay_us"]
        ).to_csv(out_dir / "table_delay_sweeps.csv", index=False)

    print(f"[CSV] wrote {out_dir / 'plot_data_combined.csv'}")
    if not worker_rows.empty:
        print(f"[CSV] wrote {out_dir / 'table_worker_scaling.csv'}")
    if not delay_rows.empty:
        print(f"[CSV] wrote {out_dir / 'table_delay_sweeps.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input grouped_tradeoff.csv files or directories containing analysis/grouped_tradeoff.csv",
    )
    ap.add_argument(
        "--out",
        default="./plots_phase3",
        help="Output directory for plots and combined CSVs",
    )
    ap.add_argument(
        "--policy",
        default="proper",
        help="Policy to use for delay-sweep plots. Default: proper",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    files = find_tradeoff_files(args.inputs)

    print("[INFO] grouped_tradeoff.csv files:")
    for f in files:
        print(f"  - {f}")

    df = read_tradeoff_files(files)
    write_selected_summary_tables(df, out_dir)

    # Worker scaling plots.
    plot_worker_scaling_metric(
        df,
        out_dir,
        metric="lc_p99_ms_mean",
        ylabel="LC p99 latency (ms)",
        filename="worker_scaling_lc_p99",
        title="LC p99 latency under increasing BE worker pressure",
    )
    plot_worker_scaling_metric(
        df,
        out_dir,
        metric="lc_p95_ms_mean",
        ylabel="LC p95 latency (ms)",
        filename="worker_scaling_lc_p95",
        title="LC p95 latency under increasing BE worker pressure",
    )
    plot_worker_scaling_metric(
        df,
        out_dir,
        metric="be_agg_thr_s_mean",
        ylabel="BE throughput (phases/s)",
        filename="worker_scaling_be_throughput",
        title="BE throughput under increasing BE worker pressure",
    )
    plot_worker_scaling_metric(
        df,
        out_dir,
        metric="overlap_weighted_ms_mean",
        ylabel="Weighted LC/BE overlap (ms)",
        filename="worker_scaling_weighted_overlap",
        title="Weighted LC/BE overlap under increasing BE worker pressure",
    )

    # Per delay-sweep plots.
    for case, worker, g in delay_groups(df, policy=args.policy):
        plot_delay_tradeoff(case, worker, g, out_dir)
        plot_delay_lc_tail(case, worker, g, out_dir)
        plot_delay_be_throughput(case, worker, g, out_dir)
        plot_delay_overlap(case, worker, g, out_dir)
        plot_delay_overlap_count(case, worker, g, out_dir)

    # Combined comparison plots for the main cases.
    plot_combined_delay_metric(
        df,
        out_dir,
        workers=["long4", "mixed4"],
        metric="lc_p99_ms_mean",
        ylabel="LC p99 latency (ms)",
        filename="combined_long_vs_mixed_lc_p99",
        title="LC p99 improvement under BE-long admission policy",
        policy=args.policy,
    )
    plot_combined_delay_metric(
        df,
        out_dir,
        workers=["long4", "mixed4"],
        metric="lc_p95_ms_mean",
        ylabel="LC p95 latency (ms)",
        filename="combined_long_vs_mixed_lc_p95",
        title="LC p95 improvement under BE-long admission policy",
        policy=args.policy,
    )
    plot_combined_delay_metric(
        df,
        out_dir,
        workers=["long4", "mixed4"],
        metric="overlap_weighted_ms_mean",
        ylabel="Weighted LC/BE overlap (ms)",
        filename="combined_long_vs_mixed_weighted_overlap",
        title="Policy reduces weighted LC/BE overlap",
        policy=args.policy,
    )
    plot_combined_delay_metric(
        df,
        out_dir,
        workers=["long4", "mixed4"],
        metric="be_agg_thr_s_mean",
        ylabel="BE throughput (phases/s)",
        filename="combined_long_vs_mixed_be_throughput",
        title="BE throughput under BE-long admission policy",
        policy=args.policy,
    )

    print()
    print(f"[DONE] plots written under: {out_dir}")


if __name__ == "__main__":
    main()
