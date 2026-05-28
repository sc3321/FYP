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

def parse_event_files(case_dir):
    events = []

    for path in case_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.name.endswith(".jsonl"):
            continue
        if path.name in ("server_stdout.log", "server_stderr.log", "config.txt", "summary.json"):
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
            events.append({
                "file": str(path),
                "line": line,
                "type": d["type"],
                "pid": int(d["pid"]),
                "phase": int(d["phase"]),
                "tid": int(d["tid"]),
                "parent": int(d["parent"]),
                "depth": int(d["depth"]),
                "t_ns": ns(d["sec"], d["nsec"]),
                "phase_type": d["phase_type"],
                "class": d["class"],
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
                "begin_line": b["line"],
                "end_line": e["line"],
            })

    return intervals

def overlap_ms(a, b):
    start = max(a["start_ns"], b["start_ns"])
    end = min(a["end_ns"], b["end_ns"])
    return max(0, end - start) / 1_000_000.0

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

def analyse_case(case_dir):
    cfg = load_config(case_dir)
    events = parse_event_files(case_dir)
    intervals = pair_intervals(events)

    lc_intervals = [x for x in intervals if x["class"] == "LC"]
    be_intervals = [x for x in intervals if x["class"] == "BE"]

    lc_durs = [x["dur_ms"] for x in lc_intervals]
    be_durs = [x["dur_ms"] for x in be_intervals]

    lc_client = parse_client(case_dir / "lc_client.jsonl")
    be_client = parse_client(case_dir / "be_client.jsonl")

    total_lc_be_overlap = sum(overlap_ms(lc, be) for lc in lc_intervals for be in be_intervals)

    row = {
        "case_dir": str(case_dir),
        "case": cfg.get("case", case_dir.name),
        "policy": cfg.get("policy", ""),
        "be_granularity": cfg.get("be_granularity", ""),
        "n_events": len(events),
        "n_intervals": len(intervals),
        "n_lc_intervals": len(lc_intervals),
        "n_be_intervals": len(be_intervals),
        "lc_be_overlap_ms": total_lc_be_overlap,

        "lc_phase_n": stats(lc_durs)["n"],
        "lc_phase_mean_ms": stats(lc_durs)["mean"],
        "lc_phase_p50_ms": stats(lc_durs)["p50"],
        "lc_phase_p95_ms": stats(lc_durs)["p95"],
        "lc_phase_p99_ms": stats(lc_durs)["p99"],

        "be_phase_n": stats(be_durs)["n"],
        "be_phase_mean_ms": stats(be_durs)["mean"],
        "be_phase_p50_ms": stats(be_durs)["p50"],
        "be_phase_p95_ms": stats(be_durs)["p95"],
        "be_phase_p99_ms": stats(be_durs)["p99"],

        "lc_client_n": stats(lc_client)["n"],
        "lc_client_mean_ms": stats(lc_client)["mean"],
        "lc_client_p50_ms": stats(lc_client)["p50"],
        "lc_client_p95_ms": stats(lc_client)["p95"],
        "lc_client_p99_ms": stats(lc_client)["p99"],

        "be_client_n": stats(be_client)["n"],
        "be_client_mean_ms": stats(be_client)["mean"],
        "be_client_p50_ms": stats(be_client)["p50"],
        "be_client_p95_ms": stats(be_client)["p95"],
        "be_client_p99_ms": stats(be_client)["p99"],
    }

    return row, intervals

def fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)

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

        with open(case_dir / "intervals.csv", "w", newline="") as f:
            fieldnames = [
                "pid", "phase", "start_ns", "end_ns", "dur_ms",
                "phase_type", "class", "begin_line", "end_line"
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(intervals)

        with open(case_dir / "summary.json", "w") as f:
            json.dump(row, f, indent=2)

    out_csv = run_dir / "summary.csv"

    fieldnames = [
        "case", "policy", "be_granularity",
        "n_events", "n_intervals", "n_lc_intervals", "n_be_intervals",
        "lc_be_overlap_ms",
        "lc_phase_n", "lc_phase_mean_ms", "lc_phase_p50_ms", "lc_phase_p95_ms", "lc_phase_p99_ms",
        "be_phase_n", "be_phase_mean_ms", "be_phase_p50_ms", "be_phase_p95_ms", "be_phase_p99_ms",
        "lc_client_n", "lc_client_mean_ms", "lc_client_p50_ms", "lc_client_p95_ms", "lc_client_p99_ms",
        "be_client_n", "be_client_mean_ms", "be_client_p50_ms", "be_client_p95_ms", "be_client_p99_ms",
        "case_dir",
    ]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
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
            f"LC client p95={fmt(r['lc_client_p95_ms'])} ms, "
            f"LC phase p95={fmt(r['lc_phase_p95_ms'])} ms, "
            f"BE client mean={fmt(r['be_client_mean_ms'])} ms, "
            f"LC/BE overlap={fmt(r['lc_be_overlap_ms'])} ms"
        )

    for name in [
        "lc_alone_none",
        "be_long_alone_none",
        "lc_be_long_none",
        "lc_be_long_policy",
        "lc_be_short_none",
        "lc_be_short_policy",
    ]:
        show(name)

if __name__ == "__main__":
    main()
