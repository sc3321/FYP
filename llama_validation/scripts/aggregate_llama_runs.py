#!/usr/bin/env python3
import csv
import math
import statistics
import sys
from pathlib import Path

KEY_COLS = ["case", "policy", "ordering", "be_granularity"]

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

    groups = {}
    for row in rows:
        key = tuple(row.get(c, "") for c in KEY_COLS)
        groups.setdefault(key, []).append(row)

    available_cols = set()
    for row in rows:
        available_cols.update(row.keys())

    numeric_cols = [c for c in IMPORTANT_NUMERIC_COLS if c in available_cols]

    for c in sorted(available_cols):
        if c in numeric_cols or c in KEY_COLS or c.startswith("_") or c == "case_dir":
            continue
        vals = [to_float(r.get(c)) for r in rows]
        if any(v is not None for v in vals):
            numeric_cols.append(c)

    long_csv = master_dir / "aggregate_long.csv"
    with open(long_csv, "w", newline="") as f:
        fieldnames = KEY_COLS + ["metric", "n", "mean", "median", "stdev", "min", "p25", "p75", "max"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for key, group_rows in sorted(groups.items()):
            key_data = dict(zip(KEY_COLS, key))
            for col in numeric_cols:
                vals = [to_float(r.get(col)) for r in group_rows]
                s = summarise(vals)

                out = {**key_data, "metric": col}
                out.update({k: fmt(v) for k, v in s.items()})
                w.writerow(out)

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
            out = dict(zip(KEY_COLS, key))
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

    # Compact report table now explicitly includes P50, P95 and P99.
    report_csv = master_dir / "aggregate_report_table.csv"
    report_metrics = [
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
    report_metrics = [m for m in report_metrics if m in available_cols]

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
            out = dict(zip(KEY_COLS, key))
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

    print(f"Loaded rows: {len(rows)}")
    print(f"Cases: {len(groups)}")
    print(f"Wrote: {long_csv}")
    print(f"Wrote: {wide_csv}")
    print(f"Wrote: {report_csv}")
    print()
    print("Compact report table:")
    with open(report_csv) as f:
        print(f.read())

if __name__ == "__main__":
    main()
