#!/usr/bin/env python3
"""
syscall_artifacts.py

Generate "scientific" artifacts from syscall sweep stats.

Input CSV must contain at least:
  variant,bytes,iterations,key,calls,seconds

You can request any metric columns that already exist in the CSV via --metrics,
e.g. usec_per_iter, calls_per_iter, usec_per_byte, calls_per_byte, pct_time_sum, ...

Core idea:
  - choose an axis to sweep: bytes OR iterations OR variant
  - fix the other two axes (recommended) using --fixed-bytes/--fixed-iterations/--fixed-variant
  - produce:
      * filtered CSV
      * trend table (long form): metric, axis_value, key, value
      * wide pivot table(s): axis rows x key columns, one file per metric
      * top-k table per axis_value (for each metric)
      * plots per metric: line plot and/or heatmap (PDF/PNG)

Example:
  python syscall_artifacts.py -i category.csv \
      --axis iterations --fixed-bytes 10000000 --fixed-variant kernels \
      --metrics usec_per_iter calls_per_iter \
      --topk 10 --outdir out --fmt pdf --dpi 300 \
      --plots line heatmap --tables filtered trend wide topk
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


AXIS_CHOICES = ["bytes", "iterations", "variant"]
PLOT_CHOICES = ["line", "heatmap"]
TABLE_CHOICES = ["filtered", "trend", "wide", "topk"]


REQUIRED_BASE_COLS = ["variant", "bytes", "iterations", "key", "calls", "seconds"]


@dataclass
class Config:
    inputs: List[str]
    axis: str
    fixed_variant: Optional[str]
    fixed_bytes: Optional[int]
    fixed_iterations: Optional[int]
    metrics: List[str]
    outdir: str
    topk: int
    include_keys: Optional[List[str]]
    exclude_keys: Optional[List[str]]
    drop_zero_metrics: bool
    fmt: str
    dpi: int
    plots: List[str]
    tables: List[str]
    logy: bool
    legend_cols: int


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Syscall sweep artifacts: tables + latexable plots.")
    p.add_argument("-i", "--input", action="append", required=True, help="CSV path (repeatable).")

    p.add_argument("--axis", choices=AXIS_CHOICES, required=True, help="Swept axis.")
    p.add_argument("--fixed-variant", default=None, help="Fix variant unless axis=variant.")
    p.add_argument("--fixed-bytes", type=int, default=None, help="Fix bytes unless axis=bytes.")
    p.add_argument("--fixed-iterations", type=int, default=None, help="Fix iterations unless axis=iterations.")

    p.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="Metric column names to analyze (must exist in CSV). "
             "E.g. usec_per_iter calls_per_iter usec_per_byte calls_per_byte",
    )

    p.add_argument("--topk", type=int, default=15,
                   help="Keep top-k keys by total magnitude across chosen metrics. 0 = keep all keys.")

    p.add_argument("--include-keys", nargs="*", default=None, help="Only include these keys.")
    p.add_argument("--exclude-keys", nargs="*", default=None, help="Exclude these keys.")
    p.add_argument("--drop-zero-metrics", action="store_true",
                   help="Drop rows where all selected metrics are 0.")

    p.add_argument("--outdir", "-o", default="artifacts_out", help="Output directory.")
    p.add_argument("--fmt", choices=["pdf", "png"], default="pdf", help="Plot output format.")
    p.add_argument("--dpi", type=int, default=300, help="Plot DPI (for png; harmless for pdf).")

    p.add_argument("--plots", nargs="*", choices=PLOT_CHOICES, default=["line", "heatmap"],
                   help="Which plots to generate.")
    p.add_argument("--tables", nargs="*", choices=TABLE_CHOICES, default=["filtered", "trend", "wide", "topk"],
                   help="Which tables to generate.")

    p.add_argument("--logy", action="store_true", help="Log-scale y-axis for line plots.")
    p.add_argument("--legend-cols", type=int, default=2, help="Legend columns for line plots.")
    args = p.parse_args()

    return Config(
        inputs=args.input,
        axis=args.axis,
        fixed_variant=args.fixed_variant,
        fixed_bytes=args.fixed_bytes,
        fixed_iterations=args.fixed_iterations,
        metrics=args.metrics,
        outdir=args.outdir,
        topk=args.topk,
        include_keys=args.include_keys,
        exclude_keys=args.exclude_keys,
        drop_zero_metrics=args.drop_zero_metrics,
        fmt=args.fmt,
        dpi=args.dpi,
        plots=args.plots,
        tables=args.tables,
        logy=args.logy,
        legend_cols=args.legend_cols,
    )


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_scientific_matplotlib_defaults() -> None:
    # Reasonable "latexable" defaults without assuming LaTeX is installed.
    plt.rcParams.update({
        "figure.figsize": (7.0, 4.2),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
    })


def load_inputs(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for pth in paths:
        df = pd.read_csv(pth)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required base columns: {missing}\nFound: {list(df.columns)}")

    df["variant"] = df["variant"].astype(str)
    df["key"] = df["key"].astype(str)

    for c in ["bytes", "iterations", "calls", "seconds"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Coerce requested metrics to numeric if they exist
    return df


def validate_metrics_exist(df: pd.DataFrame, metrics: Sequence[str]) -> None:
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(
            "These requested metrics are not present in the CSV:\n"
            f"  {missing}\n"
            "Fix: either compute them upstream (as you said you do) or remove them from --metrics."
        )


def apply_filters(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()

    if cfg.axis != "variant" and cfg.fixed_variant is not None:
        out = out[out["variant"] == cfg.fixed_variant]
    if cfg.axis != "bytes" and cfg.fixed_bytes is not None:
        out = out[out["bytes"] == cfg.fixed_bytes]
    if cfg.axis != "iterations" and cfg.fixed_iterations is not None:
        out = out[out["iterations"] == cfg.fixed_iterations]

    if cfg.include_keys:
        out = out[out["key"].isin(cfg.include_keys)]
    if cfg.exclude_keys:
        out = out[~out["key"].isin(cfg.exclude_keys)]

    # Keep only rows with axis + metrics available
    keep = [cfg.axis, "key"] + list(cfg.metrics)
    out = out.dropna(subset=keep, how="any")

    # Numeric conversion for requested metrics
    for m in cfg.metrics:
        out[m] = pd.to_numeric(out[m], errors="coerce")
    out = out.dropna(subset=list(cfg.metrics))

    return out


def maybe_topk(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.topk <= 0:
        return df

    tmp = df.copy()
    tmp["_score"] = 0.0
    for m in cfg.metrics:
        tmp["_score"] += tmp[m].fillna(0).abs()

    key_scores = tmp.groupby("key", as_index=True)["_score"].sum().sort_values(ascending=False)
    keep = set(key_scores.head(cfg.topk).index.tolist())
    tmp = tmp[tmp["key"].isin(keep)].drop(columns=["_score"])
    return tmp


def drop_zero_rows(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if not cfg.drop_zero_metrics:
        return df
    s = np.zeros(len(df), dtype=float)
    for m in cfg.metrics:
        s += df[m].fillna(0).abs().to_numpy()
    return df[s != 0]


def build_base_filename(cfg: Config) -> str:
    metric_part = "+".join(cfg.metrics)
    parts = [f"axis={cfg.axis}", f"metric={metric_part}"]

    if cfg.axis != "variant" and cfg.fixed_variant is not None:
        parts.append(f"variant={cfg.fixed_variant}")
    if cfg.axis != "bytes" and cfg.fixed_bytes is not None:
        parts.append(f"bytes={cfg.fixed_bytes}")
    if cfg.axis != "iterations" and cfg.fixed_iterations is not None:
        parts.append(f"iters={cfg.fixed_iterations}")

    if cfg.topk > 0:
        parts.append(f"topk={cfg.topk}")

    return "__".join(parts)


def make_trend_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    axis = cfg.axis

    for m in cfg.metrics:
        g = df.groupby([axis, "key"], as_index=False)[m].mean()
        g = g.rename(columns={axis: "axis_value", m: "value"})
        g.insert(0, "metric", m)
        rows.append(g)

    out = pd.concat(rows, ignore_index=True)

    if axis in ("bytes", "iterations"):
        out["axis_value_num"] = pd.to_numeric(out["axis_value"], errors="coerce")
        out = out.sort_values(["metric", "axis_value_num", "key"]).drop(columns=["axis_value_num"])
    else:
        out = out.sort_values(["metric", "axis_value", "key"])

    return out


def make_wide_pivot(df: pd.DataFrame, cfg: Config, metric: str) -> pd.DataFrame:
    pivot = (
        df.groupby([cfg.axis, "key"], as_index=False)[metric]
        .mean()
        .pivot(index=cfg.axis, columns="key", values=metric)
        .sort_index()
    )
    return pivot


def make_topk_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for xval, sub in df.groupby(cfg.axis):
        for m in cfg.metrics:
            sub_agg = sub.groupby("key", as_index=False)[m].mean().sort_values(m, ascending=False)
            top = sub_agg.head(cfg.topk if cfg.topk > 0 else min(25, len(sub_agg)))
            for _, r in top.iterrows():
                rows.append({
                    "metric": m,
                    cfg.axis: xval,
                    "key": r["key"],
                    "value": r[m],
                })
    out = pd.DataFrame(rows)
    # Sorting for readability
    if cfg.axis in ("bytes", "iterations"):
        out["_axis_num"] = pd.to_numeric(out[cfg.axis], errors="coerce")
        out = out.sort_values(["metric", "_axis_num", "value"], ascending=[True, True, False]).drop(columns=["_axis_num"])
    else:
        out = out.sort_values(["metric", cfg.axis, "value"], ascending=[True, True, False])
    return out


def axis_label(axis: str) -> str:
    if axis == "bytes":
        return "Bytes"
    if axis == "iterations":
        return "Iterations"
    return "Execution model (variant)"


def metric_label(m: str) -> str:
    # Keep labels clean + LaTeX-friendly-ish
    repl = {
        "usec_per_iter": "Syscall time per iteration (µs/iter)",
        "calls_per_iter": "Syscall calls per iteration (calls/iter)",
        "usec_per_byte": "Syscall time per byte (µs/byte)",
        "calls_per_byte": "Syscall calls per byte (calls/byte)",
        "usec_per_mb": "Syscall time per MB (µs/MB)",
        "pct_time_sum": "Share of syscall time (pct_time_sum)",
    }
    return repl.get(m, m)


def plot_line(pivot: pd.DataFrame, cfg: Config, metric: str, outpath: str) -> None:
    plt.figure()

    x = pivot.index.to_numpy()
    # If numeric axis, keep it numeric for proper scaling. Otherwise plot as categorical.
    is_numeric_axis = cfg.axis in ("bytes", "iterations")

    for col in pivot.columns:
        y = pivot[col].to_numpy(dtype=float)
        if is_numeric_axis:
            plt.plot(x, y, marker="o", linewidth=1.2, label=str(col))
        else:
            plt.plot(range(len(x)), y, marker="o", linewidth=1.2, label=str(col))

    if is_numeric_axis:
        plt.xlabel(axis_label(cfg.axis))
    else:
        plt.xlabel(axis_label(cfg.axis))
        plt.xticks(range(len(x)), [str(v) for v in x], rotation=25, ha="right")

    plt.ylabel(metric_label(metric))
    title = f"{metric} vs {cfg.axis}"
    plt.title(title)

    if cfg.logy:
        plt.yscale("log")

    # Legend outside-ish
    plt.legend(ncol=cfg.legend_cols, frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=cfg.dpi)
    plt.close()


def plot_heatmap(pivot: pd.DataFrame, cfg: Config, metric: str, outpath: str) -> None:
    data = pivot.to_numpy(dtype=float)
    x_labels = [str(c) for c in pivot.columns]
    y_labels = [str(i) for i in pivot.index]

    # Size scales with table dimensions
    w = max(7, 0.35 * len(x_labels))
    h = max(4, 0.30 * len(y_labels))
    plt.figure(figsize=(w, h))

    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, label=metric_label(metric))

    plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, rotation=60, ha="right")
    plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)

    plt.xlabel("key")
    plt.ylabel(axis_label(cfg.axis))
    plt.title(f"{metric} heatmap ({cfg.axis} × key)")

    plt.tight_layout()
    plt.savefig(outpath, dpi=cfg.dpi)
    plt.close()


def warn_if_not_fixed(cfg: Config) -> None:
    missing = []
    if cfg.axis != "variant" and cfg.fixed_variant is None:
        missing.append("--fixed-variant")
    if cfg.axis != "bytes" and cfg.fixed_bytes is None:
        missing.append("--fixed-bytes")
    if cfg.axis != "iterations" and cfg.fixed_iterations is None:
        missing.append("--fixed-iterations")
    if missing:
        print(f"[warn] For clean 1-axis sweeps, you usually want to set: {', '.join(missing)}")


def main() -> None:
    cfg = parse_args()
    ensure_outdir(cfg.outdir)
    set_scientific_matplotlib_defaults()
    warn_if_not_fixed(cfg)

    df = load_inputs(cfg.inputs)
    validate_metrics_exist(df, cfg.metrics)

    df = apply_filters(df, cfg)
    if df.empty:
        raise SystemExit("No rows after filtering. Check --axis and --fixed-* values and key filters.")

    df = maybe_topk(df, cfg)
    df = drop_zero_rows(df, cfg)
    if df.empty:
        raise SystemExit("No rows remain after topk/zero filtering.")

    base = build_base_filename(cfg)

    # ---------- TABLES ----------
    if "filtered" in cfg.tables:
        keep_cols = REQUIRED_BASE_COLS + cfg.metrics
        keep_cols = [c for c in keep_cols if c in df.columns]
        df[keep_cols].to_csv(os.path.join(cfg.outdir, f"{base}__filtered.csv"), index=False)

    if "trend" in cfg.tables:
        trend = make_trend_table(df, cfg)
        trend.to_csv(os.path.join(cfg.outdir, f"{base}__trend_long.csv"), index=False)

    if "wide" in cfg.tables:
        for m in cfg.metrics:
            pivot = make_wide_pivot(df, cfg, m)
            pivot.to_csv(os.path.join(cfg.outdir, f"{base}__wide__{m}.csv"))

    if "topk" in cfg.tables:
        topk = make_topk_table(df, cfg)
        topk.to_csv(os.path.join(cfg.outdir, f"{base}__topk_by_{cfg.axis}.csv"), index=False)

    # ---------- PLOTS ----------
    for m in cfg.metrics:
        pivot = make_wide_pivot(df, cfg, m)

        if "line" in cfg.plots:
            outpath = os.path.join(cfg.outdir, f"{base}__line__{m}.{cfg.fmt}")
            plot_line(pivot, cfg, m, outpath)

        if "heatmap" in cfg.plots:
            outpath = os.path.join(cfg.outdir, f"{base}__heatmap__{m}.{cfg.fmt}")
            plot_heatmap(pivot, cfg, m, outpath)

    print(f"Done. Output prefix:\n  {os.path.join(cfg.outdir, base)}")


if __name__ == "__main__":
    main()

