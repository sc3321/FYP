#!/usr/bin/env python3
import csv
import math
import statistics
import sys
from pathlib import Path

# Keep grouping stable across old and new runs.
# Old summaries will not have the *_np / *_concurrency fields, so they group as "".
# New continuous-LC summaries will include them.
KEY_COLS = [
    "case",
    "policy",
    "ordering",
    "be_granularity",
    "lc_server_np",
    "be_server_np",
    "lc_client_concurrency",
    "be_client_concurrency",
]

IMPORTANT_NUMERIC_COLS = [
    # client-visible latency
    "lc_client_p50_ms",
    "lc_client_p95_ms",
    "lc_client_p99_ms",
    "be_client_p50_ms",
    "be_client_p95_ms",
    "be_client_p99_ms",

    # request-level phase durations
    "lc_request_p50_ms",
    "lc_request_p95_ms",
    "lc_request_p99_ms",
    "be_request_p50_ms",
    "be_request_p95_ms",
    "be_request_p99_ms",
    "be_long_request_p50_ms",
    "be_long_request_p95_ms",
    "be_long_request_p99_ms",
    "be_short_request_p50_ms",
    "be_short_request_p95_ms",
    "be_short_request_p99_ms",

    # decode-level phase durations
    "lc_decode_p50_ms",
    "lc_decode_p95_ms",
    "lc_decode_p99_ms",
    "be_decode_p50_ms",
    "be_decode_p95_ms",
    "be_decode_p99_ms",
    "be_long_decode_p50_ms",
    "be_long_decode_p95_ms",
    "be_long_decode_p99_ms",
    "be_short_decode_p50_ms",
    "be_short_decode_p95_ms",
    "be_short_decode_p99_ms",

    # overlaps
    "request_lc_be_overlap_ms",
    "request_lc_be_long_overlap_ms",
    "request_lc_be_short_overlap_ms",
    "decode_lc_be_overlap_ms",
    "decode_lc_be_long_overlap_ms",
    "decode_lc_be_short_overlap_ms",

    # counts
    "n_events",
    "n_intervals",
    "n_request_intervals",
    "n_decode_intervals",
    "n_lc_request",
    "n_be_request",
    "n_lc_decode",
    "n_be_decode",

    # policy-counter diagnostics from the updated single-run parser
    "policy_counter_samples",
    "policy_active_lc_gt0_samples",
    "policy_active_lc_eq0_samples",
    "policy_active_lc_max",
    "policy_active_belong_gt0_samples",
    "policy_active_belong_max",
    "policy_active_lc_zero_and_belong_gt0_samples",
    "policy_checks_final",
    "policy_belong_saw_lc_active_final",
    "policy_belong_imm_admit_final",
    "policy_belong_delay_admit_final",
    "policy_belong_throttle_count_final",
    "policy_belong_wait_us_final",
    "policy_configured_delay_us",
    "policy_belong_imm_increments_while_active_lc_zero",
    "policy_belong_delay_increments_while_active_lc_gt0",
]

REPORT_METRICS = [
    # client latency
    "lc_client_p50_ms",
    "lc_client_p95_ms",
    "lc_client_p99_ms",
    "be_client_p50_ms",
    "be_client_p95_ms",
    "be_client_p99_ms",

    # request-level duration
    "lc_request_p50_ms",
    "lc_request_p95_ms",
    "lc_request_p99_ms",
    "be_request_p50_ms",
    "be_request_p95_ms",
    "be_request_p99_ms",
    "be_long_request_p50_ms",
    "be_long_request_p95_ms",
    "be_long_request_p99_ms",
    "be_short_request_p50_ms",
    "be_short_request_p95_ms",
    "be_short_request_p99_ms",

    # decode-level duration
    "lc_decode_p50_ms",
    "lc_decode_p95_ms",
    "lc_decode_p99_ms",
    "be_decode_p50_ms",
    "be_decode_p95_ms",
    "be_decode_p99_ms",
    "be_long_decode_p50_ms",
    "be_long_decode_p95_ms",
    "be_long_decode_p99_ms",
    "be_short_decode_p50_ms",
    "be_short_decode_p95_ms",
    "be_short_decode_p99_ms",

    # overlaps
    "request_lc_be_overlap_ms",
    "request_lc_be_long_overlap_ms",
    "request_lc_be_short_overlap_ms",
    "decode_lc_be_overlap_ms",
    "decode_lc_be_long_overlap_ms",
    "decode_lc_be_short_overlap_ms",
]

POLICY_REPORT_METRICS = [
    "policy_counter_samples",
    "policy_active_lc_gt0_samples",
    "policy_active_lc_eq0_samples",
    "policy_active_lc_max",
    "policy_active_belong_gt0_samples",
    "policy_active_belong_max",
    "policy_active_lc_zero_and_belong_gt0_samples",
    "policy_checks_final",
    "policy_belong_saw_lc_active_final",
    "policy_belong_imm_admit_final",
    "policy_belong_delay_admit_final",
    "policy_belong_throttle_count_final",
    "policy_belong_wait_us_final",
    "policy_configured_delay_us",
    "policy_belong_imm_increments_while_active_lc_zero",
    "policy_belong_delay_increments_while_active_lc_gt0",
]

DIRECT_COMPARE_PAIRS = [
    ("Serialized LC BE-long: CAP vs NONE", "lc_be_long_none", "lc_be_long_policy"),
    ("Serialized LC BE-short: CAP vs NONE", "lc_be_short_none", "lc_be_short_policy"),
    ("LC-first BE-long: CAP vs NONE", "lc_first_be_long_none", "lc_first_be_long_policy"),
    ("Continuous LC BE-long: CAP vs NONE", "lc_cont_be_long_none", "lc_cont_be_long_policy"),
]


def to_float(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == "":
        return None
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except ValueError:
        return None


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = math.ceil((p / 100.0) * len(xs)) - 1
    k = max(0, min(k, len(xs) - 1))
    return xs[k]


def summarise(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {
            "n": 0,
            "mean": "",
            "median": "",
            "stdev": "",
            "min": "",
            "p25": "",
            "p75": "",
            "max": "",
        }

    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "p25": percentile(xs, 25),
        "p75": percentile(xs, 75),
        "max": max(xs),
    }


def fmt(x):
    if x == "":
        return ""
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def load_rows(master_dir):
    rows = []

    for run_dir in sorted(master_dir.glob("run_[0-9][0-9][0-9]")):
        summary = run_dir / "summary.csv"
        if not summary.exists():
            continue

        with open(summary, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_run"] = run_dir.name
                rows.append(row)

    return rows


def present_key(row):
    return tuple(row.get(c, "") for c in KEY_COLS)


def build_groups(rows):
    groups = {}
    for row in rows:
        key = present_key(row)
        groups.setdefault(key, []).append(row)
    return groups


def group_key_dict(key):
    return dict(zip(KEY_COLS, key))


def available_columns(rows):
    cols = set()
    for row in rows:
        cols.update(row.keys())
    return cols


def discover_numeric_cols(rows, available_cols):
    numeric_cols = [c for c in IMPORTANT_NUMERIC_COLS if c in available_cols]

    # Auto-include additional numeric columns emitted by newer parser versions.
    for c in sorted(available_cols):
        if c in numeric_cols or c in KEY_COLS or c.startswith("_") or c == "case_dir":
            continue
        vals = [to_float(r.get(c)) for r in rows]
        if any(v is not None for v in vals):
            numeric_cols.append(c)

    return numeric_cols


def write_long_table(master_dir, groups, numeric_cols):
    long_csv = master_dir / "aggregate_long.csv"

    with open(long_csv, "w", newline="") as f:
        fieldnames = KEY_COLS + ["metric", "n", "mean", "median", "stdev", "min", "p25", "p75", "max"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for key, group_rows in sorted(groups.items()):
            key_data = group_key_dict(key)
            for col in numeric_cols:
                vals = [to_float(r.get(col)) for r in group_rows]
                s = summarise(vals)

                out = {**key_data, "metric": col}
                out.update({k: fmt(v) for k, v in s.items()})
                w.writerow(out)

    return long_csv


def write_wide_table(master_dir, groups, numeric_cols):
    wide_csv = master_dir / "aggregate_wide.csv"

    with open(wide_csv, "w", newline="") as f:
        fieldnames = KEY_COLS + ["valid_runs"]
        for col in numeric_cols:
            fieldnames += [
                f"{col}_mean",
                f"{col}_median",
                f"{col}_stdev",
                f"{col}_p25",
                f"{col}_p75",
                f"{col}_min",
                f"{col}_max",
            ]

        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for key, group_rows in sorted(groups.items()):
            out = group_key_dict(key)
            out["valid_runs"] = len({r["_run"] for r in group_rows})

            for col in numeric_cols:
                vals = [to_float(r.get(col)) for r in group_rows]
                s = summarise(vals)
                out[f"{col}_mean"] = fmt(s["mean"])
                out[f"{col}_median"] = fmt(s["median"])
                out[f"{col}_stdev"] = fmt(s["stdev"])
                out[f"{col}_p25"] = fmt(s["p25"])
                out[f"{col}_p75"] = fmt(s["p75"])
                out[f"{col}_min"] = fmt(s["min"])
                out[f"{col}_max"] = fmt(s["max"])

            w.writerow(out)

    return wide_csv


def write_report_table(master_dir, groups, available_cols):
    report_csv = master_dir / "aggregate_report_table.csv"
    report_metrics = [m for m in REPORT_METRICS if m in available_cols]

    with open(report_csv, "w", newline="") as f:
        fieldnames = KEY_COLS + ["valid_runs"]
        for m in report_metrics:
            fieldnames += [
                f"{m}_median",
                f"{m}_mean",
                f"{m}_stdev",
                f"{m}_p25",
                f"{m}_p75",
                f"{m}_min",
                f"{m}_max",
            ]

        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for key, group_rows in sorted(groups.items()):
            out = group_key_dict(key)
            out["valid_runs"] = len({r["_run"] for r in group_rows})

            for m in report_metrics:
                vals = [to_float(r.get(m)) for r in group_rows]
                s = summarise(vals)
                out[f"{m}_median"] = fmt(s["median"])
                out[f"{m}_mean"] = fmt(s["mean"])
                out[f"{m}_stdev"] = fmt(s["stdev"])
                out[f"{m}_p25"] = fmt(s["p25"])
                out[f"{m}_p75"] = fmt(s["p75"])
                out[f"{m}_min"] = fmt(s["min"])
                out[f"{m}_max"] = fmt(s["max"])

            w.writerow(out)

    return report_csv


def write_policy_report_table(master_dir, groups, available_cols):
    policy_csv = master_dir / "aggregate_policy_table.csv"
    metrics = [m for m in POLICY_REPORT_METRICS if m in available_cols]

    with open(policy_csv, "w", newline="") as f:
        fieldnames = KEY_COLS + ["valid_runs"]
        for m in metrics:
            fieldnames += [
                f"{m}_median",
                f"{m}_mean",
                f"{m}_stdev",
                f"{m}_p25",
                f"{m}_p75",
                f"{m}_min",
                f"{m}_max",
            ]

        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for key, group_rows in sorted(groups.items()):
            out = group_key_dict(key)
            out["valid_runs"] = len({r["_run"] for r in group_rows})

            for m in metrics:
                vals = [to_float(r.get(m)) for r in group_rows]
                s = summarise(vals)
                out[f"{m}_median"] = fmt(s["median"])
                out[f"{m}_mean"] = fmt(s["mean"])
                out[f"{m}_stdev"] = fmt(s["stdev"])
                out[f"{m}_p25"] = fmt(s["p25"])
                out[f"{m}_p75"] = fmt(s["p75"])
                out[f"{m}_min"] = fmt(s["min"])
                out[f"{m}_max"] = fmt(s["max"])

            w.writerow(out)

    return policy_csv


def median_for_case(groups, case_name, metric):
    vals = []
    for key, group_rows in groups.items():
        key_data = group_key_dict(key)
        if key_data.get("case") != case_name:
            continue
        vals.extend(to_float(r.get(metric)) for r in group_rows)
    s = summarise(vals)
    return s["median"], s["mean"], s["n"]


def write_delta_table(master_dir, groups, available_cols):
    delta_csv = master_dir / "aggregate_policy_deltas.csv"

    metrics = [
        "lc_client_p50_ms",
        "lc_client_p95_ms",
        "lc_client_p99_ms",
        "lc_request_p95_ms",
        "lc_decode_p95_ms",
        "be_client_p95_ms",
        "request_lc_be_overlap_ms",
        "decode_lc_be_overlap_ms",
        "policy_belong_imm_admit_final",
        "policy_belong_delay_admit_final",
        "policy_belong_throttle_count_final",
        "policy_belong_wait_us_final",
    ]
    metrics = [m for m in metrics if m in available_cols]

    with open(delta_csv, "w", newline="") as f:
        fieldnames = [
            "comparison",
            "none_case",
            "policy_case",
            "metric",
            "none_median",
            "policy_median",
            "delta_median",
            "pct_delta_median",
            "none_mean",
            "policy_mean",
            "delta_mean",
            "pct_delta_mean",
            "none_n",
            "policy_n",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for comparison, none_case, policy_case in DIRECT_COMPARE_PAIRS:
            for metric in metrics:
                none_median, none_mean, none_n = median_for_case(groups, none_case, metric)
                policy_median, policy_mean, policy_n = median_for_case(groups, policy_case, metric)

                def delta(a, b):
                    if a == "" or b == "":
                        return ""
                    return b - a

                def pct(a, b):
                    if a == "" or b == "" or a == 0:
                        return ""
                    return 100.0 * (b - a) / a

                out = {
                    "comparison": comparison,
                    "none_case": none_case,
                    "policy_case": policy_case,
                    "metric": metric,
                    "none_median": fmt(none_median),
                    "policy_median": fmt(policy_median),
                    "delta_median": fmt(delta(none_median, policy_median)),
                    "pct_delta_median": fmt(pct(none_median, policy_median)),
                    "none_mean": fmt(none_mean),
                    "policy_mean": fmt(policy_mean),
                    "delta_mean": fmt(delta(none_mean, policy_mean)),
                    "pct_delta_mean": fmt(pct(none_mean, policy_mean)),
                    "none_n": fmt(none_n),
                    "policy_n": fmt(policy_n),
                }
                w.writerow(out)

    return delta_csv


def print_file(path, title):
    print()
    print(title)
    print("=" * len(title))
    try:
        with open(path) as f:
            print(f.read())
    except Exception as e:
        print(f"Could not print {path}: {e}")


def main():
    if len(sys.argv) != 2:
        print("usage: aggregate_llama_repeated_runs.py REPEATED_RUN_DIR", file=sys.stderr)
        sys.exit(1)

    master_dir = Path(sys.argv[1])
    if not master_dir.exists():
        print(f"ERROR: directory does not exist: {master_dir}", file=sys.stderr)
        sys.exit(1)

    rows = load_rows(master_dir)
    if not rows:
        print(f"ERROR: no rows found under {master_dir}/run_*/summary.csv", file=sys.stderr)
        sys.exit(1)

    groups = build_groups(rows)
    available_cols = available_columns(rows)
    numeric_cols = discover_numeric_cols(rows, available_cols)

    long_csv = write_long_table(master_dir, groups, numeric_cols)
    wide_csv = write_wide_table(master_dir, groups, numeric_cols)
    report_csv = write_report_table(master_dir, groups, available_cols)
    policy_csv = write_policy_report_table(master_dir, groups, available_cols)
    delta_csv = write_delta_table(master_dir, groups, available_cols)

    print(f"Loaded rows: {len(rows)}")
    print(f"Cases/groups: {len(groups)}")
    print(f"Wrote: {long_csv}")
    print(f"Wrote: {wide_csv}")
    print(f"Wrote: {report_csv}")
    print(f"Wrote: {policy_csv}")
    print(f"Wrote: {delta_csv}")

    print_file(delta_csv, "Policy delta table")
    print_file(report_csv, "Compact report table")
    if any(c in available_cols for c in POLICY_REPORT_METRICS):
        print_file(policy_csv, "Policy counter table")


if __name__ == "__main__":
    main()

