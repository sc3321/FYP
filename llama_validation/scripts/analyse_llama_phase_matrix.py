#!/usr/bin/env python3
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

EVENT_RE = re.compile(
    r"Event type = (?P<type>BEGIN|END): "
    r"PhaseId:\[(?P<pid>\d+),\s*(?P<phase>\d+)\],"
    r"Thread Id:\s*(?P<tid>\d+), "
    r"parent_id:\s*(?P<parent>\d+), "
    r"depth:\s*(?P<depth>\d+),\s*"
    r"Timestamp:\s*(?P<sec>\d+)\s*s\s*(?P<nsec>\d+)\s*ns, "
    r"phase type:\s*\((?P<phase_type>[^)]+)\), "
    r"workload class:\s*(?P<class>\w+)"
    r"(?:,\s*granularity:\s*(?P<granularity>\w+))?"
)

def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = math.ceil((p / 100.0) * len(xs)) - 1
    k = max(0, min(k, len(xs) - 1))
    return xs[k]

def stats(xs):
    if not xs:
        return {
            "n": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "p50": percentile(xs, 50),
        "p95": percentile(xs, 95),
        "p99": percentile(xs, 99),
        "min": min(xs),
        "max": max(xs),
    }

def ns(sec, nsec):
    return int(sec) * 1_000_000_000 + int(nsec)

def norm_phase_type(x):
    return (x or "UNKNOWN").strip().upper()

def norm_class(x):
    return (x or "UNKNOWN").strip().upper()

def norm_granularity(x):
    if x is None:
        return "UNKNOWN"
    return x.strip().upper()

def infer_granularity_from_config(cls, cfg):
    cls = norm_class(cls)
    if cls == "LC":
        return "SHORT"
    if cls == "BE":
        return cfg.get("be_granularity", "UNKNOWN").strip().upper()
    return "UNKNOWN"

def parse_event_files(case_dir, cfg):
    events = []

    for path in case_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.name.endswith(".jsonl"):
            continue
        if path.name in ("server_stdout.log", "server_stderr.log", "config.txt", "summary.json", "intervals.csv"):
            continue
        if path.name.endswith(".json") and "responses_" in str(path):
            continue

        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue

        for line in text.splitlines():
            m = EVENT_RE.search(line)
            if not m:
                continue

            d = m.groupdict()

            cls = norm_class(d["class"])
            gran = norm_granularity(d.get("granularity"))

            # Backward compatibility for old logs before granularity was added.
            if gran == "UNKNOWN":
                gran = infer_granularity_from_config(cls, cfg)

            events.append({
                "file": str(path),
                "line": line,
                "type": d["type"].upper(),
                "pid": int(d["pid"]),
                "phase": int(d["phase"]),
                "tid": int(d["tid"]),
                "parent": int(d["parent"]),
                "depth": int(d["depth"]),
                "t_ns": ns(d["sec"], d["nsec"]),
                "phase_type": norm_phase_type(d["phase_type"]),
                "class": cls,
                "granularity": gran,
            })

    return sorted(events, key=lambda e: e["t_ns"])

def pair_intervals(events):
    active = {}
    intervals = []

    for e in events:
        key = (e["pid"], e["phase"])

        if e["type"] == "BEGIN":
            active[key] = e

        elif e["type"] == "END":
            b = active.pop(key, None)
            if b is None:
                continue

            intervals.append({
                "pid": b["pid"],
                "phase": b["phase"],
                "start_ns": b["t_ns"],
                "end_ns": e["t_ns"],
                "dur_ms": (e["t_ns"] - b["t_ns"]) / 1_000_000.0,
                "phase_type": b["phase_type"],
                "class": b["class"],
                "granularity": b["granularity"],
                "begin_line": b["line"],
                "end_line": e["line"],
            })

    return intervals

def overlap_ms(a, b):
    start = max(a["start_ns"], b["start_ns"])
    end = min(a["end_ns"], b["end_ns"])
    return max(0, end - start) / 1_000_000.0

def total_overlap(left, right):
    return sum(overlap_ms(a, b) for a in left for b in right)

def parse_client(path):
    vals = []
    if not path.exists():
        return vals
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            vals.append(json.loads(line)["latency_ms"])
        except Exception:
            pass
    return vals

def load_config(case_dir):
    cfg = {}
    p = case_dir / "config.txt"
    if not p.exists():
        return cfg
    for line in p.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg

def filt(intervals, phase_type=None, cls=None, granularity=None):
    out = intervals
    if phase_type is not None:
        phase_type = phase_type.upper()
        out = [x for x in out if x["phase_type"] == phase_type]
    if cls is not None:
        cls = cls.upper()
        out = [x for x in out if x["class"] == cls]
    if granularity is not None:
        granularity = granularity.upper()
        out = [x for x in out if x["granularity"] == granularity]
    return out

def add_stats(row, prefix, intervals):
    durs = [x["dur_ms"] for x in intervals]
    s = stats(durs)
    row[f"{prefix}_n"] = s["n"]
    row[f"{prefix}_mean_ms"] = s["mean"]
    row[f"{prefix}_p50_ms"] = s["p50"]
    row[f"{prefix}_p95_ms"] = s["p95"]
    row[f"{prefix}_p99_ms"] = s["p99"]
    row[f"{prefix}_min_ms"] = s["min"]
    row[f"{prefix}_max_ms"] = s["max"]

def add_client_stats(row, prefix, vals):
    s = stats(vals)
    row[f"{prefix}_n"] = s["n"]
    row[f"{prefix}_mean_ms"] = s["mean"]
    row[f"{prefix}_p50_ms"] = s["p50"]
    row[f"{prefix}_p95_ms"] = s["p95"]
    row[f"{prefix}_p99_ms"] = s["p99"]
    row[f"{prefix}_min_ms"] = s["min"]
    row[f"{prefix}_max_ms"] = s["max"]

def analyse_case(case_dir):
    cfg = load_config(case_dir)
    events = parse_event_files(case_dir, cfg)
    intervals = pair_intervals(events)

    request_lc = filt(intervals, "LLAMA_REQUEST", "LC")
    request_be = filt(intervals, "LLAMA_REQUEST", "BE")
    request_be_long = filt(intervals, "LLAMA_REQUEST", "BE", "LONG")
    request_be_short = filt(intervals, "LLAMA_REQUEST", "BE", "SHORT")

    decode_lc = filt(intervals, "LLAMA_DECODE", "LC")
    decode_be = filt(intervals, "LLAMA_DECODE", "BE")
    decode_be_long = filt(intervals, "LLAMA_DECODE", "BE", "LONG")
    decode_be_short = filt(intervals, "LLAMA_DECODE", "BE", "SHORT")

    all_lc = filt(intervals, None, "LC")
    all_be = filt(intervals, None, "BE")

    lc_client = parse_client(case_dir / "lc_client.jsonl")
    be_client = parse_client(case_dir / "be_client.jsonl")

    row = {
        "case_dir": str(case_dir),
        "case": cfg.get("case", case_dir.name),
        "policy": cfg.get("policy", ""),
        "ordering": cfg.get("ordering", ""),
        "be_granularity": cfg.get("be_granularity", ""),
        "n_events": len(events),
        "n_intervals": len(intervals),

        "n_request_intervals": len(filt(intervals, "LLAMA_REQUEST")),
        "n_decode_intervals": len(filt(intervals, "LLAMA_DECODE")),

        "n_lc_intervals_all": len(all_lc),
        "n_be_intervals_all": len(all_be),

        "n_lc_request": len(request_lc),
        "n_be_request": len(request_be),
        "n_be_long_request": len(request_be_long),
        "n_be_short_request": len(request_be_short),

        "n_lc_decode": len(decode_lc),
        "n_be_decode": len(decode_be),
        "n_be_long_decode": len(decode_be_long),
        "n_be_short_decode": len(decode_be_short),

        "request_lc_be_overlap_ms": total_overlap(request_lc, request_be),
        "request_lc_be_long_overlap_ms": total_overlap(request_lc, request_be_long),
        "request_lc_be_short_overlap_ms": total_overlap(request_lc, request_be_short),

        "decode_lc_be_overlap_ms": total_overlap(decode_lc, decode_be),
        "decode_lc_be_long_overlap_ms": total_overlap(decode_lc, decode_be_long),
        "decode_lc_be_short_overlap_ms": total_overlap(decode_lc, decode_be_short),

        # Backward-compatible name, but now it means request-level overlap only.
        "lc_be_overlap_ms": total_overlap(request_lc, request_be),
    }

    add_stats(row, "lc_request", request_lc)
    add_stats(row, "be_request", request_be)
    add_stats(row, "be_long_request", request_be_long)
    add_stats(row, "be_short_request", request_be_short)

    add_stats(row, "lc_decode", decode_lc)
    add_stats(row, "be_decode", decode_be)
    add_stats(row, "be_long_decode", decode_be_long)
    add_stats(row, "be_short_decode", decode_be_short)

    add_client_stats(row, "lc_client", lc_client)
    add_client_stats(row, "be_client", be_client)

    return row, intervals

def fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)

def write_intervals(case_dir, intervals):
    with open(case_dir / "intervals.csv", "w", newline="") as f:
        fieldnames = [
            "pid", "phase", "start_ns", "end_ns", "dur_ms",
            "phase_type", "class", "granularity", "begin_line", "end_line"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(intervals)

def main():
    if len(sys.argv) != 2:
        print("usage: analyse_llama_phase_matrix.py RUN_DIR")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"run dir does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    case_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir()])
    rows = []

    for case_dir in case_dirs:
        row, intervals = analyse_case(case_dir)
        rows.append(row)

        write_intervals(case_dir, intervals)

        with open(case_dir / "summary.json", "w") as f:
            json.dump(row, f, indent=2)

    out_csv = run_dir / "summary.csv"

    fieldnames = [
        "case", "policy", "ordering", "be_granularity",
        "n_events", "n_intervals", "n_request_intervals", "n_decode_intervals",
        "n_lc_request", "n_be_request", "n_be_long_request", "n_be_short_request",
        "n_lc_decode", "n_be_decode", "n_be_long_decode", "n_be_short_decode",

        "request_lc_be_overlap_ms",
        "request_lc_be_long_overlap_ms",
        "request_lc_be_short_overlap_ms",
        "decode_lc_be_overlap_ms",
        "decode_lc_be_long_overlap_ms",
        "decode_lc_be_short_overlap_ms",

        "lc_request_n", "lc_request_mean_ms", "lc_request_p50_ms", "lc_request_p95_ms", "lc_request_p99_ms",
        "be_request_n", "be_request_mean_ms", "be_request_p50_ms", "be_request_p95_ms", "be_request_p99_ms",
        "be_long_request_n", "be_long_request_mean_ms", "be_long_request_p50_ms", "be_long_request_p95_ms", "be_long_request_p99_ms",
        "be_short_request_n", "be_short_request_mean_ms", "be_short_request_p50_ms", "be_short_request_p95_ms", "be_short_request_p99_ms",

        "lc_decode_n", "lc_decode_mean_ms", "lc_decode_p50_ms", "lc_decode_p95_ms", "lc_decode_p99_ms",
        "be_decode_n", "be_decode_mean_ms", "be_decode_p50_ms", "be_decode_p95_ms", "be_decode_p99_ms",
        "be_long_decode_n", "be_long_decode_mean_ms", "be_long_decode_p50_ms", "be_long_decode_p95_ms", "be_long_decode_p99_ms",
        "be_short_decode_n", "be_short_decode_mean_ms", "be_short_decode_p50_ms", "be_short_decode_p95_ms", "be_short_decode_p99_ms",

        "lc_client_n", "lc_client_mean_ms", "lc_client_p50_ms", "lc_client_p95_ms", "lc_client_p99_ms",
        "be_client_n", "be_client_mean_ms", "be_client_p50_ms", "be_client_p95_ms", "be_client_p99_ms",

        # Backward-compatible old column name.
        "lc_be_overlap_ms",

        "case_dir",
    ]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote: {out_csv}")
    print()
    print("Summary:")
    print(",".join(fieldnames))
    for r in rows:
        print(",".join(fmt(r.get(k)) for k in fieldnames))

    print()
    print("Most important comparisons:")
    by_case = {r["case"]: r for r in rows}

    def show(name):
        r = by_case.get(name)
        if not r:
            return
        print(
            f"{name}: "
            f"LC client p95={fmt(r.get('lc_client_p95_ms'))} ms, "
            f"LC request p95={fmt(r.get('lc_request_p95_ms'))} ms, "
            f"LC decode p95={fmt(r.get('lc_decode_p95_ms'))} ms, "
            f"BE client p95={fmt(r.get('be_client_p95_ms'))} ms, "
            f"BE request p95={fmt(r.get('be_request_p95_ms'))} ms, "
            f"BE decode p95={fmt(r.get('be_decode_p95_ms'))} ms, "
            f"request overlap={fmt(r.get('request_lc_be_overlap_ms'))} ms, "
            f"decode overlap={fmt(r.get('decode_lc_be_overlap_ms'))} ms"
        )

    for name in [
        "lc_alone_none",
        "be_long_alone_none",
        "lc_be_long_none",
        "lc_be_long_policy",
        "lc_be_short_none",
        "lc_be_short_policy",
        "lc_first_be_long_none",
        "lc_first_be_long_policy",
    ]:
        show(name)

if __name__ == "__main__":
    main()
