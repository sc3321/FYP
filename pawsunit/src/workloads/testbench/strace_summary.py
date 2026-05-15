#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
from collections import defaultdict

LINE_RE = re.compile(
    r"^\s*(?P<ts>\d+\.\d+)\s+"
    r"(?P<syscall>[a-zA-Z_][a-zA-Z0-9_]*)\(.*"
    r"<(?P<dur>\d+\.\d+)>"
)


def parse_strace_file(path):
    counts = defaultdict(int)
    total_time = defaultdict(float)

    with open(path, "r", errors="replace") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            syscall = m.group("syscall")
            dur = float(m.group("dur"))
            counts[syscall] += 1
            total_time[syscall] += dur

    return counts, total_time


def collect_strace(run_dir):
    counts = defaultdict(int)
    total_time = defaultdict(float)

    for path in run_dir.glob("*strace*"):
        if path.is_file():
            c, t = parse_strace_file(path)
            for k, v in c.items():
                counts[k] += v
            for k, v in t.items():
                total_time[k] += v

    return counts, total_time


def print_summary(run_name, counts, total_time):
    print()
    print("#" * 80)
    print(f"STRACE SUMMARY: {run_name}")
    print("#" * 80)

    rows = []
    for syscall, count in counts.items():
        rows.append((total_time[syscall], count, syscall))

    rows.sort(reverse=True)

    print(f"{'syscall':<24} {'count':>10} {'total_s':>12} {'avg_us':>12}")
    print("-" * 62)

    for total, count, syscall in rows:
        avg_us = (total / count) * 1e6 if count else 0.0
        print(f"{syscall:<24} {count:>10} {total:>12.6f} {avg_us:>12.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--runs", nargs="*", default=[
        "lc_alone_stable_r1",
        "lc_vs_4_be_long_r1",
        "lc_vs_1_be_chunked_r1",
    ])
    args = ap.parse_args()

    root = Path(args.output_dir)

    for run in args.runs:
        run_dir = root / run
        if not run_dir.exists():
            print(f"[WARN] missing {run_dir}")
            continue
        counts, total_time = collect_strace(run_dir)
        print_summary(run, counts, total_time)


if __name__ == "__main__":
    main()
