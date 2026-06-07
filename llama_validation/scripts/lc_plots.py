#!/usr/bin/env python3
"""
lc_plots_ch7.py

Generate the revised Chapter 7 figure set for the LC/BE phase-aware
policy evaluation.

The script keeps the original inputs where possible, but changes the
figure set to remove redundancy and make the cross-layer story clearer.

Inputs:
    --os-per-rep-csv
        Path to lc_syscall_compare_per_rep.csv from the OS/eBPF analysis.

    --os-replicates-root
        Root of the OS/eBPF matrix run. Used to find ebpf_events.jsonl files
        for the policy-signature nanosleep plot.

    --app-replicates-root
        Root of the application-level matrix run. Use the run without eBPF
        tracing attached so the application metrics are not polluted by tracer
        overhead. If omitted, falls back to --os-replicates-root.

    --app-summary-csv
        Optional CSV containing already-computed application p99 metrics.
        The parser is intentionally flexible and recognises columns such as
        lc_client_p99_ms, lc_request_p99_ms, lc_decode_p99_ms,
        be_client_p99_ms, client_p99_ms, request_p99_ms, decode_p99_ms.
        If omitted, the script scans per-replicate case directories for JSONL
        and CSV files and computes p99 values where raw samples are available.

    --bench-jsonl
        Optional explicit cap-side ebpf_events.jsonl for Figure 1.

    --none-bench-jsonl
        Optional explicit none-side ebpf_events.jsonl for Figure 1.

    --out-dir
        Output directory.

Outputs:
    fig1_policy_signature_nanosleep.png
    fig2_os_futex_reshape_jk.png
    fig3_app_latency_stack_jk.png
    fig4_lc_be_tradeoff_jk.png
    fig5_cross_layer_case_matrix.png
    figure_data_summary.csv

Design principle:
    Each figure should support one sentence in Chapter 7:
      1. The policy is host-observable.
      2. The LC futex distribution is reshaped under sustained contention.
      3. The application improvement starts at decode and propagates upward.
      4. LC improvement is paid for by BE latency.
      5. The cross-layer pattern distinguishes sustained, brief, and null cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm


# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLOR_NONE = "#4C566A"      # muted slate
COLOR_CAP = "#2F80ED"       # clear blue
COLOR_OS = "#7B61FF"        # purple
COLOR_APP = "#27AE60"       # green
COLOR_BE = "#D35400"        # orange
COLOR_NEUTRAL = "#333333"
COLOR_LIGHT = "#E5E7EB"
MARKER_NONE = "o"
MARKER_CAP = "s"


# -----------------------------------------------------------------------------
# Case definitions
# -----------------------------------------------------------------------------

CASE_PAIRS = {
    "C/D": {
        "label": "C/D\nsequential",
        "comparison": "be_first sequential (C/D)",
        "none": "caseC_lc_be_long_none",
        "cap": "caseD_lc_be_long_policy",
    },
    "E/F": {
        "label": "E/F\nnegative control",
        "comparison": "be_short neg-control (E/F)",
        "none": "caseE_lc_be_short_none",
        "cap": "caseF_lc_be_short_policy",
    },
    "J/K": {
        "label": "J/K\ncontinuous",
        "comparison": "lc_cont continuous (J/K)",
        "none": "caseJ_lc_cont_be_long_none",
        "cap": "caseK_lc_cont_be_long_policy",
    },
}


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def ok(msg: str) -> None:
    print(f"[ok] {msg}", file=sys.stderr)


def skip(msg: str) -> None:
    print(f"[skip] {msg}", file=sys.stderr)


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if math.isfinite(float(x)):
            return float(x)
        return None
    s = str(x).strip()
    if not s or s.lower() in {"none", "nan", "null", "na", "n/a"}:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.percentile(vals, p, method="linear"))


def mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.mean(vals))


def pct_delta(none_value: Optional[float], cap_value: Optional[float]) -> Optional[float]:
    if none_value is None or cap_value is None or none_value == 0:
        return None
    return (cap_value - none_value) / none_value * 100.0


def format_delta(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.1f}%"


def read_csv_rows(path: Optional[str | Path]) -> List[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        warn(f"missing csv: {p}")
        return []
    with p.open("r", newline="") as fh:
        return list(csv.DictReader(fh))


def list_rep_dirs(root: Optional[str | Path]) -> List[Path]:
    if not root:
        return []
    p = Path(root)
    if not p.exists():
        warn(f"missing replicates root: {p}")
        return []
    reps = [
        d for d in p.iterdir()
        if d.is_dir() and re.match(r"^(rep|run_)?\d+$", d.name)
    ]
    # Also support rep01 / run_001 style.
    reps += [
        d for d in p.iterdir()
        if d.is_dir() and re.match(r"^(rep|run_)\d+$", d.name)
    ]
    # De-duplicate while preserving sorted order.
    reps = sorted(set(reps), key=lambda d: d.name)
    return reps


def case_letter(case_dirname: str) -> Optional[str]:
    m = re.match(r"case([A-Z])", case_dirname)
    return m.group(1) if m else None


def case_matches(row_value: str, case_dirname: str) -> bool:
    if not row_value:
        return False
    rv = str(row_value).strip()
    if rv == case_dirname:
        return True
    letter = case_letter(case_dirname)
    if letter and rv.upper() == letter:
        return True
    if letter and rv.lower() == f"case{letter.lower()}":
        return True
    return rv in case_dirname or case_dirname in rv


def numeric_values(xs: Iterable[Optional[float]]) -> List[float]:
    out = []
    for x in xs:
        v = safe_float(x)
        if v is not None:
            out.append(v)
    return out


def save_summary_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# OS/eBPF data helpers
# -----------------------------------------------------------------------------

def os_per_rep_for(
    rows: List[dict],
    comparison: str,
    kind: str,
    cohort: str,
    thread_role: str = "app",
) -> List[Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]]:
    """Return (rep, none_p99_us, cap_p99_us, none_p95_us, cap_p95_us, delta_pct)."""
    out = []
    for r in rows:
        if r.get("comparison") != comparison:
            continue
        if r.get("kind") != kind:
            continue
        if r.get("cohort") != cohort:
            continue
        if r.get("thread_role") != thread_role:
            continue
        out.append((
            str(r.get("rep", "")),
            safe_float(r.get("none_p99_us")),
            safe_float(r.get("cap_p99_us")),
            safe_float(r.get("none_p95_us")),
            safe_float(r.get("cap_p95_us")),
            safe_float(r.get("p99_delta_pct")),
        ))
    out.sort(key=lambda x: x[0])
    return out


def os_delta_series(os_rows: List[dict], comparison: str) -> List[float]:
    vals = []
    for r in os_rows:
        if r.get("comparison") != comparison:
            continue
        if r.get("kind") != "futex":
            continue
        if r.get("cohort") != "woken":
            continue
        if r.get("thread_role") != "app":
            continue
        v = safe_float(r.get("p99_delta_pct"))
        if v is not None:
            vals.append(v)
    return vals


# -----------------------------------------------------------------------------
# Application data helpers
# -----------------------------------------------------------------------------

@dataclass
class MetricResult:
    metric: str
    kind: str
    case_dirname: str
    samples_ms: List[float]
    per_rep_p99_ms: Dict[str, float]
    summary_p99_ms: Optional[float] = None

    @property
    def p99_ms(self) -> Optional[float]:
        if self.samples_ms:
            return percentile(self.samples_ms, 99)
        if self.summary_p99_ms is not None:
            return self.summary_p99_ms
        vals = list(self.per_rep_p99_ms.values())
        if vals:
            return mean(vals)
        return None

    @property
    def n_samples(self) -> int:
        return len(self.samples_ms)

    @property
    def n_reps(self) -> int:
        return len(self.per_rep_p99_ms)


def flatten_json(prefix: str, obj, out: Dict[str, object]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flatten_json(key, v, out)
    elif isinstance(obj, list):
        # Lists are ignored for metric extraction; they are rarely scalar timings.
        return
    else:
        out[prefix.lower()] = obj


def read_jsonl(path: Path) -> Iterator[dict]:
    try:
        with path.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def read_json(path: Path) -> Iterator[dict]:
    try:
        with path.open("r") as fh:
            obj = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(obj, dict):
        yield obj
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item


def event_kind(ev: dict, source: Path) -> Optional[str]:
    flat: Dict[str, object] = {}
    flatten_json("", ev, flat)
    for key in ("kind", "role", "class", "workload", "workload_kind", "request_kind", "phase_kind"):
        for fk, value in flat.items():
            if fk.endswith(key):
                s = str(value).strip().lower()
                if s in {"lc", "latency_critical", "latency-critical", "latencycritical"}:
                    return "lc"
                if s in {"be", "best_effort", "best-effort", "besteffort", "background"}:
                    return "be"
    name = source.name.lower()
    stem = source.stem.lower()
    if re.search(r"(^|[_\-.])lc([_\-.]|$)", stem):
        return "lc"
    if re.search(r"(^|[_\-.])be([_\-.]|$)", stem):
        return "be"
    if "lc_client" in name:
        return "lc"
    if "be_client" in name:
        return "be"
    return None


def event_matches_kind(ev: dict, source: Path, wanted_kind: str) -> bool:
    k = event_kind(ev, source)
    # If the file/event carries an explicit LC/BE label, respect it.
    if k is not None:
        return k == wanted_kind
    # Otherwise accept the event. This permits server-side summaries that are
    # already LC-only but do not carry a kind field.
    return True


def unit_to_ms(key: str, value: float) -> float:
    k = key.lower()
    if k.endswith("_ns") or k.endswith(".ns") or "duration_ns" in k or "latency_ns" in k:
        return value / 1e6
    if k.endswith("_us") or k.endswith(".us") or "duration_us" in k or "latency_us" in k:
        return value / 1e3
    if k.endswith("_s") or k.endswith(".s") or k in {"seconds", "duration_s", "latency_s"}:
        return value * 1e3
    return value


def pick_numeric(flat: Dict[str, object], candidate_keys: Sequence[str]) -> Optional[float]:
    # Exact suffix match first.
    for wanted in candidate_keys:
        wanted_l = wanted.lower()
        for key, value in flat.items():
            if key == wanted_l or key.endswith("." + wanted_l) or key.endswith("_" + wanted_l):
                v = safe_float(value)
                if v is not None:
                    return unit_to_ms(key, v)
    # Then weaker containment match.
    for wanted in candidate_keys:
        wanted_l = wanted.lower()
        for key, value in flat.items():
            if wanted_l in key:
                v = safe_float(value)
                if v is not None:
                    return unit_to_ms(key, v)
    return None


def extract_sample_ms(ev: dict, source: Path, metric: str) -> Optional[float]:
    """
    Extract one raw latency sample in ms from a JSON event.

    This is deliberately permissive because the runner evolved over time.
    It handles common names from client logs, server summaries, and phase logs.
    """
    flat: Dict[str, object] = {}
    flatten_json("", ev, flat)

    metric_l = metric.lower()

    # If the event explicitly names a phase, use generic duration fields only
    # when the phase matches the requested metric.
    phase_text = " ".join(
        str(v).lower() for k, v in flat.items()
        if any(tok in k for tok in ("phase", "name", "metric", "label"))
    )

    if metric_l == "client":
        # Prefer explicitly client/end-to-end fields. The generic latency_ms
        # key is accepted mainly for lc_client.jsonl/be_client.jsonl logs; using
        # it indiscriminately would risk mistaking phase latencies for client
        # latency in server-side event files.
        v = pick_numeric(flat, [
            "client_latency_ms", "client_ms", "e2e_ms", "end_to_end_ms",
            "elapsed_ms", "wall_ms", "lc_client_ms", "be_client_ms",
        ])
        if v is not None:
            return v
        source_name = source.name.lower()
        if "client" in source_name or "request" not in phase_text and "decode" not in phase_text:
            return pick_numeric(flat, ["latency_ms"])
        return None

    if metric_l == "request":
        v = pick_numeric(flat, [
            "request_ms", "request_latency_ms", "server_request_ms",
            "server_latency_ms", "server_ms", "server_total_ms",
            "request_total_ms", "lc_request_ms", "be_request_ms",
        ])
        if v is not None:
            return v
        if "request" in phase_text:
            return pick_numeric(flat, ["duration_ms", "latency_ms", "elapsed_ms", "dur_ms", "duration_us", "dur_us", "duration_ns", "dur_ns"])
        return None

    if metric_l == "decode":
        v = pick_numeric(flat, [
            "decode_ms", "decode_latency_ms", "decode_token_ms",
            "per_token_decode_ms", "token_decode_ms", "decode_time_ms",
            "lc_decode_ms", "be_decode_ms",
        ])
        if v is not None:
            return v
        if "decode" in phase_text:
            return pick_numeric(flat, ["duration_ms", "latency_ms", "elapsed_ms", "dur_ms", "duration_us", "dur_us", "duration_ns", "dur_ns"])
        return None

    return None


def extract_summary_p99_ms(ev: dict, source: Path, metric: str, kind: str) -> Optional[float]:
    flat: Dict[str, object] = {}
    flatten_json("", ev, flat)
    prefixes = [f"{kind}_{metric}", metric]
    candidates = []
    for p in prefixes:
        candidates.extend([
            f"{p}_p99_ms", f"{p}_p99", f"p99_{p}_ms", f"p99_{p}",
            f"{p}.p99_ms", f"{p}.p99",
        ])
    # Avoid treating raw latency_ms as a summary.
    return pick_numeric(flat, candidates)


def scan_case_metric_from_files(
    app_root: Optional[str | Path],
    case_dirname: str,
    metric: str,
    kind: str,
) -> MetricResult:
    result = MetricResult(metric=metric, kind=kind, case_dirname=case_dirname, samples_ms=[], per_rep_p99_ms={})
    reps = list_rep_dirs(app_root)
    if not reps:
        return result

    for rep_dir in reps:
        case_dir = rep_dir / case_dirname
        if not case_dir.exists():
            continue

        rep_samples: List[float] = []
        rep_summary_vals: List[float] = []

        json_files = list(case_dir.rglob("*.jsonl")) + list(case_dir.rglob("*.json"))
        # Prefer files whose name suggests the metric/kind but still scan all.
        json_files = sorted(set(json_files), key=lambda p: (
            0 if kind in p.name.lower() else 1,
            0 if metric in p.name.lower() else 1,
            p.name,
        ))

        for path in json_files:
            events = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
            for ev in events:
                if not event_matches_kind(ev, path, kind):
                    continue
                sample = extract_sample_ms(ev, path, metric)
                if sample is not None:
                    rep_samples.append(sample)
                summary = extract_summary_p99_ms(ev, path, metric, kind)
                if summary is not None:
                    rep_summary_vals.append(summary)

        # CSV files occasionally contain per-replicate summaries.
        for path in sorted(case_dir.rglob("*.csv")):
            rows = read_csv_rows(path)
            for row in rows:
                if row_matches_kind(row, kind):
                    sample = extract_sample_from_csv_row(row, metric, kind)
                    if sample is not None:
                        rep_samples.append(sample)
                    summary = extract_summary_from_csv_row(row, metric, kind)
                    if summary is not None:
                        rep_summary_vals.append(summary)

        if rep_samples:
            result.samples_ms.extend(rep_samples)
            p99 = percentile(rep_samples, 99)
            if p99 is not None:
                result.per_rep_p99_ms[rep_dir.name] = p99
        elif rep_summary_vals:
            # If the replicate only has summary p99 rows, average duplicates.
            result.per_rep_p99_ms[rep_dir.name] = float(np.mean(rep_summary_vals))

    return result


def row_matches_kind(row: dict, kind: str) -> bool:
    for key in ("kind", "role", "class", "workload", "request_kind", "workload_kind"):
        if key in row and row[key]:
            s = str(row[key]).strip().lower()
            if kind == "lc" and s in {"lc", "latency-critical", "latency_critical"}:
                return True
            if kind == "be" and s in {"be", "best-effort", "best_effort", "background"}:
                return True
            return False
    return True


def extract_sample_from_csv_row(row: dict, metric: str, kind: str) -> Optional[float]:
    lower = {str(k).lower(): v for k, v in row.items()}
    candidates = {
        "client": ["latency_ms", "client_latency_ms", "client_ms", "elapsed_ms", f"{kind}_client_ms"],
        "request": ["request_ms", "request_latency_ms", "server_ms", f"{kind}_request_ms"],
        "decode": ["decode_ms", "decode_latency_ms", "decode_token_ms", f"{kind}_decode_ms"],
    }.get(metric, [])
    for key in candidates:
        if key in lower:
            v = safe_float(lower[key])
            if v is not None:
                return unit_to_ms(key, v)
    return None


def extract_summary_from_csv_row(row: dict, metric: str, kind: str) -> Optional[float]:
    lower = {str(k).lower(): v for k, v in row.items()}
    candidates = [
        f"{kind}_{metric}_p99_ms", f"{kind}_{metric}_p99",
        f"{metric}_p99_ms", f"{metric}_p99",
        f"p99_{kind}_{metric}_ms", f"p99_{metric}_ms",
    ]
    for key in candidates:
        if key in lower:
            v = safe_float(lower[key])
            if v is not None:
                return unit_to_ms(key, v)
    return None


def metric_from_summary_csv(
    summary_csv: Optional[str | Path],
    case_dirname: str,
    metric: str,
    kind: str,
) -> MetricResult:
    result = MetricResult(metric=metric, kind=kind, case_dirname=case_dirname, samples_ms=[], per_rep_p99_ms={})
    rows = read_csv_rows(summary_csv)
    if not rows:
        return result

    vals = []
    for row in rows:
        # Case matching: look for common case columns; if absent, keep row.
        case_cols = [c for c in row.keys() if c.lower() in {"case", "case_name", "case_dir", "case_dirname", "scenario"}]
        if case_cols:
            if not any(case_matches(str(row.get(c, "")), case_dirname) for c in case_cols):
                continue
        if not row_matches_kind(row, kind):
            continue
        summary = extract_summary_from_csv_row(row, metric, kind)
        if summary is not None:
            rep = str(row.get("rep") or row.get("replicate") or row.get("run") or len(vals) + 1)
            result.per_rep_p99_ms[rep] = summary
            vals.append(summary)

    if vals:
        result.summary_p99_ms = float(np.mean(vals))
    return result


def combine_metric_results(primary: MetricResult, fallback: MetricResult) -> MetricResult:
    """Prefer summary CSV values when present; fill missing from scanned files."""
    if primary.p99_ms is not None or primary.per_rep_p99_ms:
        if not primary.samples_ms and fallback.samples_ms:
            primary.samples_ms = fallback.samples_ms
        for rep, val in fallback.per_rep_p99_ms.items():
            primary.per_rep_p99_ms.setdefault(rep, val)
        return primary
    return fallback


def app_metric(
    app_root: Optional[str | Path],
    app_summary_csv: Optional[str | Path],
    case_dirname: str,
    metric: str,
    kind: str = "lc",
) -> MetricResult:
    from_csv = metric_from_summary_csv(app_summary_csv, case_dirname, metric, kind)
    from_files = scan_case_metric_from_files(app_root, case_dirname, metric, kind)
    return combine_metric_results(from_csv, from_files)


def paired_app_deltas(
    app_root: Optional[str | Path],
    app_summary_csv: Optional[str | Path],
    case_none: str,
    case_cap: str,
    metric: str,
    kind: str = "lc",
) -> List[float]:
    none = app_metric(app_root, app_summary_csv, case_none, metric, kind)
    cap = app_metric(app_root, app_summary_csv, case_cap, metric, kind)
    deltas: List[float] = []

    for rep, none_p99 in none.per_rep_p99_ms.items():
        cap_p99 = cap.per_rep_p99_ms.get(rep)
        d = pct_delta(none_p99, cap_p99)
        if d is not None:
            deltas.append(d)

    # If paired replicate summaries are unavailable, fall back to one pooled delta.
    if not deltas:
        d = pct_delta(none.p99_ms, cap.p99_ms)
        if d is not None:
            deltas.append(d)
    return deltas


# -----------------------------------------------------------------------------
# eBPF nanosleep signature helpers
# -----------------------------------------------------------------------------

def find_default_jsonl(root: Optional[str | Path], preferred_cases: Sequence[str]) -> Optional[Path]:
    if not root:
        return None
    for rep_dir in list_rep_dirs(root):
        for case in preferred_cases:
            p = rep_dir / case / "ebpf_events.jsonl"
            if p.exists():
                return p
        # Fallback: fuzzy match in this replicate.
        for p in sorted(rep_dir.rglob("ebpf_events.jsonl")):
            name = str(p).lower()
            if any(case.lower() in name for case in preferred_cases):
                return p
    return None


def nanosleep_durations_ms(path: Optional[str | Path]) -> List[float]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        warn(f"missing ebpf jsonl: {p}")
        return []
    vals: List[float] = []
    for ev in read_jsonl(p):
        kind = str(ev.get("kind", "")).lower()
        if kind not in {"clock_nanosleep", "nanosleep"}:
            continue
        dur_ns = safe_float(ev.get("dur_ns"))
        dur_us = safe_float(ev.get("dur_us"))
        dur_ms = safe_float(ev.get("dur_ms"))
        if dur_ns is not None:
            vals.append(dur_ns / 1e6)
        elif dur_us is not None:
            vals.append(dur_us / 1e3)
        elif dur_ms is not None:
            vals.append(dur_ms)
    return vals


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def annotate_delta(ax, x: float, y: float, text: str) -> None:
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": COLOR_LIGHT, "alpha": 0.95},
    )


def scatter_two_groups(
    ax,
    none_vals: Sequence[float],
    cap_vals: Sequence[float],
    ylabel: str,
    left_label: str = "none",
    right_label: str = "cap",
    seed: int = 0,
) -> Optional[float]:
    none_vals = numeric_values(none_vals)
    cap_vals = numeric_values(cap_vals)
    if not none_vals or not cap_vals:
        return None

    rng = np.random.default_rng(seed)
    x_none = rng.normal(0, 0.035, size=len(none_vals))
    x_cap = rng.normal(1, 0.035, size=len(cap_vals))

    ax.scatter(x_none, none_vals, color=COLOR_NONE, marker=MARKER_NONE,
               s=26, alpha=0.85, edgecolor="black", linewidth=0.25, zorder=3)
    ax.scatter(x_cap, cap_vals, color=COLOR_CAP, marker=MARKER_CAP,
               s=26, alpha=0.85, edgecolor="black", linewidth=0.25, zorder=3)

    mn = mean(none_vals)
    mc = mean(cap_vals)
    if mn is not None:
        ax.plot([-0.22, 0.22], [mn, mn], color=COLOR_NONE, linewidth=2.0, zorder=4)
    if mc is not None:
        ax.plot([0.78, 1.22], [mc, mc], color=COLOR_CAP, linewidth=2.0, zorder=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([left_label, right_label])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=COLOR_LIGHT, linewidth=0.6, alpha=0.8)
    return pct_delta(mn, mc)


def plot_delta_strip(
    ax,
    data: List[List[float]],
    labels: List[str],
    colors: List[str],
    ylabel: str,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    tick_labels: List[str] = []
    for i, vals in enumerate(data):
        vals = numeric_values(vals)
        if not vals:
            tick_labels.append(f"{labels[i]}\nn/a")
            continue
        x = rng.normal(i, 0.04, size=len(vals))
        ax.scatter(x, vals, color=colors[i], s=28, alpha=0.85,
                   edgecolor="black", linewidth=0.25, zorder=3)
        m = mean(vals)
        if m is not None:
            ax.plot([i - 0.25, i + 0.25], [m, m], color="black", linewidth=2.0, zorder=4)
            tick_labels.append(f"{labels[i]}\nmean {m:+.1f}%")
        else:
            tick_labels.append(f"{labels[i]}\nn/a")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=COLOR_LIGHT, linewidth=0.6, alpha=0.8)


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def fig1_policy_signature_nanosleep(
    none_jsonl: Optional[str | Path],
    cap_jsonl: Optional[str | Path],
    out_dir: Path,
    summary_rows: List[dict],
) -> None:
    none_vals = [v for v in nanosleep_durations_ms(none_jsonl) if 0 <= v <= 20.0]
    cap_vals = [v for v in nanosleep_durations_ms(cap_jsonl) if 0 <= v <= 20.0]

    if not none_vals and not cap_vals:
        skip("fig 1: no nanosleep durations found")
        return

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bins = np.linspace(0, 20, 81)
    if none_vals:
        ax.hist(none_vals, bins=bins, histtype="step", linewidth=1.4,
                color=COLOR_NONE, label=f"none, n={len(none_vals)}")
    if cap_vals:
        ax.hist(cap_vals, bins=bins, histtype="stepfilled", alpha=0.35,
                color=COLOR_CAP, edgecolor=COLOR_CAP, linewidth=0.8,
                label=f"cap, n={len(cap_vals)}")

    ax.axvspan(4, 8, color=COLOR_BE, alpha=0.12, label="4-8 ms backoff band")
    ax.set_xlabel("BE clock_nanosleep duration (ms)")
    ax.set_ylabel("event count")
    ax.set_xlim(0, 20)
    ax.grid(axis="y", color=COLOR_LIGHT, linewidth=0.6, alpha=0.8)
    ax.legend(loc="upper right", frameon=False)

    out = out_dir / "fig1_policy_signature_nanosleep.png"
    fig.savefig(out)
    plt.close(fig)
    ok(f"fig 1: wrote {out}")

    summary_rows.append({
        "figure": "fig1_policy_signature_nanosleep",
        "metric": "clock_nanosleep_count_0_20ms",
        "none": len(none_vals),
        "cap": len(cap_vals),
    })


def fig2_os_futex_reshape_jk(os_rows: List[dict], out_dir: Path, summary_rows: List[dict]) -> None:
    pairs = os_per_rep_for(
        os_rows,
        comparison=CASE_PAIRS["J/K"]["comparison"],
        kind="futex",
        cohort="woken",
        thread_role="app",
    )
    if not pairs:
        skip("fig 2: no J/K futex wake-cohort rows in OS CSV")
        return

    none_p95_us = [p[3] for p in pairs if p[3] is not None]
    cap_p95_us = [p[4] for p in pairs if p[4] is not None]
    none_p99_s = [p[1] / 1e6 for p in pairs if p[1] is not None]
    cap_p99_s = [p[2] / 1e6 for p in pairs if p[2] is not None]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.45))

    d95 = scatter_two_groups(
        axes[0], none_p95_us, cap_p95_us,
        ylabel="LC futex wake p95 (us)",
        left_label="J\nnone", right_label="K\ncap", seed=11,
    )
    axes[0].set_title("Body")
    annotate_delta(axes[0], 0.98, 0.96, f"mean delta {format_delta(d95)}")

    d99 = scatter_two_groups(
        axes[1], none_p99_s, cap_p99_s,
        ylabel="LC futex wake p99 (s)",
        left_label="J\nnone", right_label="K\ncap", seed=12,
    )
    axes[1].set_title("Tail")
    annotate_delta(axes[1], 0.98, 0.96, f"mean delta {format_delta(d99)}")

    legend_elements = [
        Line2D([0], [0], marker=MARKER_NONE, color="w", label="case J / none",
               markerfacecolor=COLOR_NONE, markeredgecolor="black", markersize=6),
        Line2D([0], [0], marker=MARKER_CAP, color="w", label="case K / cap",
               markerfacecolor=COLOR_CAP, markeredgecolor="black", markersize=6),
    ]
    axes[1].legend(handles=legend_elements, loc="lower right", frameon=False)

    out = out_dir / "fig2_os_futex_reshape_jk.png"
    fig.savefig(out)
    plt.close(fig)
    ok(f"fig 2: wrote {out}")

    summary_rows.extend([
        {"figure": "fig2_os_futex_reshape_jk", "metric": "os_futex_woken_p95_us", "none": mean(none_p95_us), "cap": mean(cap_p95_us), "delta_pct": d95, "n_none": len(none_p95_us), "n_cap": len(cap_p95_us)},
        {"figure": "fig2_os_futex_reshape_jk", "metric": "os_futex_woken_p99_s", "none": mean(none_p99_s), "cap": mean(cap_p99_s), "delta_pct": d99, "n_none": len(none_p99_s), "n_cap": len(cap_p99_s)},
    ])


def fig3_app_latency_stack_jk(
    app_root: Optional[str | Path],
    app_summary_csv: Optional[str | Path],
    out_dir: Path,
    summary_rows: List[dict],
) -> None:
    pair = CASE_PAIRS["J/K"]
    metrics = ["decode", "request", "client"]
    labels = ["Decode", "Request", "Client"]
    colors = ["#8E44AD", COLOR_APP, COLOR_CAP]

    delta_series = []
    p99_pairs = []
    for metric in metrics:
        none = app_metric(app_root, app_summary_csv, pair["none"], metric, kind="lc")
        cap = app_metric(app_root, app_summary_csv, pair["cap"], metric, kind="lc")
        ds = paired_app_deltas(app_root, app_summary_csv, pair["none"], pair["cap"], metric, kind="lc")
        delta_series.append(ds)
        p99_pairs.append((none, cap))

    if not any(delta_series):
        skip("fig 3: no LC app decode/request/client metrics found for J/K")
        return

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    plot_delta_strip(
        ax,
        data=delta_series,
        labels=labels,
        colors=colors,
        ylabel="LC p99 delta under cap (%)",
        seed=21,
    )
    ax.set_ylim(auto=True)
    # Make improvement direction visually explicit without using a long title.
    ymin, ymax = ax.get_ylim()
    ax.text(0.01, 0.05, "lower is better for LC", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.5, color=COLOR_NEUTRAL)

    out = out_dir / "fig3_app_latency_stack_jk.png"
    fig.savefig(out)
    plt.close(fig)
    ok(f"fig 3: wrote {out}")

    for metric, (none, cap), ds in zip(metrics, p99_pairs, delta_series):
        summary_rows.append({
            "figure": "fig3_app_latency_stack_jk",
            "metric": f"lc_{metric}_p99_ms",
            "none": none.p99_ms,
            "cap": cap.p99_ms,
            "delta_pct": mean(ds),
            "n_none_samples": none.n_samples,
            "n_cap_samples": cap.n_samples,
            "n_paired_reps_or_points": len(ds),
        })


def fig4_lc_be_tradeoff_jk(
    app_root: Optional[str | Path],
    app_summary_csv: Optional[str | Path],
    out_dir: Path,
    summary_rows: List[dict],
) -> None:
    pair = CASE_PAIRS["J/K"]
    specs = [
        ("LC client", "client", "lc", COLOR_APP),
        ("BE client", "client", "be", COLOR_BE),
    ]
    delta_series = []
    labels = []
    colors = []
    p99_pairs = []

    for label, metric, kind, color in specs:
        none = app_metric(app_root, app_summary_csv, pair["none"], metric, kind=kind)
        cap = app_metric(app_root, app_summary_csv, pair["cap"], metric, kind=kind)
        ds = paired_app_deltas(app_root, app_summary_csv, pair["none"], pair["cap"], metric, kind=kind)
        labels.append(label)
        colors.append(color)
        delta_series.append(ds)
        p99_pairs.append((none, cap))

    if not any(delta_series):
        skip("fig 4: no LC/BE client metrics found for J/K")
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.45))
    plot_delta_strip(
        ax,
        data=delta_series,
        labels=labels,
        colors=colors,
        ylabel="client p99 delta under cap (%)",
        seed=31,
    )
    ax.text(0.01, 0.05, "negative protects LC; positive penalises BE",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5,
            color=COLOR_NEUTRAL)

    out = out_dir / "fig4_lc_be_tradeoff_jk.png"
    fig.savefig(out)
    plt.close(fig)
    ok(f"fig 4: wrote {out}")

    for label, (none, cap), ds in zip(labels, p99_pairs, delta_series):
        summary_rows.append({
            "figure": "fig4_lc_be_tradeoff_jk",
            "metric": label.lower().replace(" ", "_") + "_p99_ms",
            "none": none.p99_ms,
            "cap": cap.p99_ms,
            "delta_pct": mean(ds),
            "n_none_samples": none.n_samples,
            "n_cap_samples": cap.n_samples,
            "n_paired_reps_or_points": len(ds),
        })


def fig5_cross_layer_case_matrix(
    os_rows: List[dict],
    app_root: Optional[str | Path],
    app_summary_csv: Optional[str | Path],
    out_dir: Path,
    summary_rows: List[dict],
) -> None:
    row_specs = [
        ("OS futex\np99", "os", "futex"),
        ("App decode\np99", "app", "decode"),
        ("App request\np99", "app", "request"),
        ("App client\np99", "app", "client"),
    ]
    col_keys = ["C/D", "E/F", "J/K"]
    matrix = np.full((len(row_specs), len(col_keys)), np.nan)
    cell_counts = np.zeros_like(matrix)

    for j, key in enumerate(col_keys):
        pair = CASE_PAIRS[key]
        for i, (_, layer, metric) in enumerate(row_specs):
            if layer == "os":
                vals = os_delta_series(os_rows, pair["comparison"])
            else:
                vals = paired_app_deltas(app_root, app_summary_csv, pair["none"], pair["cap"], metric, kind="lc")
            m = mean(vals)
            if m is not None:
                matrix[i, j] = m
                cell_counts[i, j] = len(vals)
                summary_rows.append({
                    "figure": "fig5_cross_layer_case_matrix",
                    "case_pair": key,
                    "metric": row_specs[i][0].replace("\n", " "),
                    "delta_pct": m,
                    "n_points": len(vals),
                })

    if np.all(np.isnan(matrix)):
        skip("fig 5: no cross-layer deltas available")
        return

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    finite = matrix[np.isfinite(matrix)]
    max_abs = max(5.0, float(np.nanmax(np.abs(finite)))) if finite.size else 5.0
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(col_keys)))
    ax.set_xticklabels([CASE_PAIRS[k]["label"] for k in col_keys])
    ax.set_yticks(np.arange(len(row_specs)))
    ax.set_yticklabels([r[0] for r in row_specs])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:+.1f}%\n(n={int(cell_counts[i, j])})",
                        ha="center", va="center", fontsize=8.5, color="black")
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8.5, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("p99 delta under cap (%)")
    ax.set_xlabel("case pair")
    ax.set_ylabel("measurement layer")

    out = out_dir / "fig5_cross_layer_case_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    ok(f"fig 5: wrote {out}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate revised Chapter 7 LC/BE policy figures.")
    ap.add_argument("--os-per-rep-csv", required=True,
                    help="Path to lc_syscall_compare_per_rep.csv from the OS/eBPF analysis.")
    ap.add_argument("--os-replicates-root", required=True,
                    help="OS/eBPF matrix root, used for policy-signature nanosleep events.")
    ap.add_argument("--app-replicates-root", default=None,
                    help="Application-level matrix root without eBPF tracing. Falls back to OS root if omitted.")
    ap.add_argument("--app-summary-csv", default=None,
                    help="Optional CSV containing app p99 columns such as lc_client_p99_ms, lc_request_p99_ms, lc_decode_p99_ms, be_client_p99_ms.")
    ap.add_argument("--bench-jsonl", default=None,
                    help="Optional cap-side ebpf_events.jsonl for Figure 1.")
    ap.add_argument("--none-bench-jsonl", default=None,
                    help="Optional none-side ebpf_events.jsonl for Figure 1.")
    ap.add_argument("--out-dir", default="./figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os_rows = read_csv_rows(args.os_per_rep_csv)
    if not os_rows:
        print(f"[error] no rows in {args.os_per_rep_csv}", file=sys.stderr)
        sys.exit(1)

    app_root = args.app_replicates_root or args.os_replicates_root
    if app_root == args.os_replicates_root:
        warn("--app-replicates-root not given; application plots may include eBPF tracer overhead.")

    cap_jsonl = args.bench_jsonl or find_default_jsonl(
        args.os_replicates_root,
        ["caseD_lc_be_long_policy", "caseK_lc_cont_be_long_policy", "caseF_lc_be_short_policy"],
    )
    none_jsonl = args.none_bench_jsonl or find_default_jsonl(
        args.os_replicates_root,
        ["caseC_lc_be_long_none", "caseJ_lc_cont_be_long_none", "caseE_lc_be_short_none"],
    )

    if not cap_jsonl:
        warn("could not auto-find a cap-side ebpf_events.jsonl for Figure 1")
    if not none_jsonl:
        warn("could not auto-find a none-side ebpf_events.jsonl for Figure 1; Figure 1 will show cap only if available")

    summary_rows: List[dict] = []

    fig1_policy_signature_nanosleep(none_jsonl, cap_jsonl, out_dir, summary_rows)
    fig2_os_futex_reshape_jk(os_rows, out_dir, summary_rows)
    fig3_app_latency_stack_jk(app_root, args.app_summary_csv, out_dir, summary_rows)
    fig4_lc_be_tradeoff_jk(app_root, args.app_summary_csv, out_dir, summary_rows)
    fig5_cross_layer_case_matrix(os_rows, app_root, args.app_summary_csv, out_dir, summary_rows)

    summary_path = out_dir / "figure_data_summary.csv"
    save_summary_csv(summary_path, summary_rows)
    if summary_rows:
        ok(f"wrote {summary_path}")

    print(f"\nFigures written to: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()

