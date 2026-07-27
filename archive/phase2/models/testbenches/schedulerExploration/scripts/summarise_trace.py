#!/usr/bin/env python3
import os
import re
import sys
from collections import defaultdict

CATEGORY_MAP = {
    "futex": "sync",
    "poll": "wait",
    "ppoll": "wait",
    "epoll_wait": "wait",
    "epoll_pwait": "wait",
    "select": "wait",
    "pselect6": "wait",
    "ioctl": "driver",
    "mmap": "vm",
    "munmap": "vm",
    "mprotect": "vm",
    "brk": "vm",
    "mremap": "vm",
    "madvise": "vm",
    "mlock": "vm",
    "munlock": "vm",
    "nanosleep": "sleep",
    "clock_nanosleep": "sleep",
    "clone": "threading",
    "clone3": "threading",
    "pthread_create": "threading",
    "sched_yield": "sched",
    "sched_setaffinity": "sched",
    "sched_getaffinity": "sched",
    "write": "io",
    "read": "io",
    "open": "io",
    "openat": "io",
    "close": "io",
}

FOCUS_SYSCALLS = [
    "futex",
    "poll",
    "ppoll",
    "epoll_wait",
    "epoll_pwait",
    "ioctl",
    "mmap",
    "munmap",
    "mprotect",
    "brk",
    "mremap",
    "madvise",
    "nanosleep",
    "clock_nanosleep",
    "clone",
    "clone3",
    "sched_yield",
]

LINE_RE = re.compile(
    r"""
    ^
    (?:\[\w+\]\s+)?                 # optional strace prefix
    (?:(\d+(?:\.\d+)?)\s+)?         # optional absolute timestamp
    ([a-zA-Z_][a-zA-Z0-9_]*)        # syscall name
    \(
    .*?
    (?:=\s+[-\dxa-fA-F?]+.*?)?      # return value area
    (?:<([\d.]+)>)?                 # optional duration
    \s*$
    """,
    re.VERBOSE,
)

UNFINISHED_RE = re.compile(r'^\s*(?:\[\w+\]\s+)?(?:(\d+(?:\.\d+)?)\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\(.*<unfinished \.\.\.>\s*$')
RESUMED_RE = re.compile(r'^\s*(?:\[\w+\]\s+)?(?:(\d+(?:\.\d+)?)\s+)?<\.\.\. ([a-zA-Z_][a-zA-Z0-9_]*) resumed>.*(?:<([\d.]+)>)?\s*$')

def category_of(syscall: str) -> str:
    return CATEGORY_MAP.get(syscall, "other")

def summarise_file(path: str):
    syscall_counts = defaultdict(int)
    syscall_time = defaultdict(float)
    category_counts = defaultdict(int)
    category_time = defaultdict(float)

    total_lines = 0
    parsed_lines = 0
    unfinished = 0
    resumed = 0

    with open(path, "r", errors="replace") as f:
        for line in f:
            total_lines += 1
            line = line.rstrip("\n")

            if "<unfinished ...>" in line:
                m = UNFINISHED_RE.match(line)
                if m:
                    syscall = m.group(2)
                    syscall_counts[syscall] += 1
                    category_counts[category_of(syscall)] += 1
                    unfinished += 1
                    parsed_lines += 1
                    continue

            if "<... " in line and " resumed>" in line:
                m = RESUMED_RE.match(line)
                if m:
                    syscall = m.group(2)
                    dur = float(m.group(3)) if m.group(3) else 0.0
                    syscall_time[syscall] += dur
                    category_time[category_of(syscall)] += dur
                    resumed += 1
                    parsed_lines += 1
                    continue

            m = LINE_RE.match(line)
            if not m:
                continue

            syscall = m.group(2)
            dur = float(m.group(3)) if m.group(3) else 0.0

            syscall_counts[syscall] += 1
            syscall_time[syscall] += dur
            category_counts[category_of(syscall)] += 1
            category_time[category_of(syscall)] += dur
            parsed_lines += 1

    return {
        "path": path,
        "total_lines": total_lines,
        "parsed_lines": parsed_lines,
        "unfinished": unfinished,
        "resumed": resumed,
        "syscall_counts": dict(syscall_counts),
        "syscall_time": dict(syscall_time),
        "category_counts": dict(category_counts),
        "category_time": dict(category_time),
    }

def role_guess(stats):
    sc = stats["syscall_counts"]
    st = stats["syscall_time"]

    futex_t = st.get("futex", 0.0)
    wait_t = sum(st.get(x, 0.0) for x in ["poll", "ppoll", "epoll_wait", "epoll_pwait"])
    ioctl_t = st.get("ioctl", 0.0)
    vm_t = sum(st.get(x, 0.0) for x in ["mmap", "munmap", "mprotect", "brk", "mremap", "madvise"])
    sleep_t = sum(st.get(x, 0.0) for x in ["nanosleep", "clock_nanosleep"])

    ioctl_c = sc.get("ioctl", 0)
    futex_c = sc.get("futex", 0)
    wait_c = sum(sc.get(x, 0) for x in ["poll", "ppoll", "epoll_wait", "epoll_pwait"])
    vm_c = sum(sc.get(x, 0) for x in ["mmap", "munmap", "mprotect", "brk", "mremap", "madvise"])

    if ioctl_t > max(futex_t, wait_t, vm_t, sleep_t) or ioctl_c > max(futex_c, wait_c, vm_c, 1):
        return "likely submission/driver-heavy thread"
    if futex_t > max(ioctl_t, wait_t, vm_t, sleep_t):
        return "likely sync/wake thread"
    if wait_t > max(ioctl_t, futex_t, vm_t, sleep_t):
        return "likely completion-wait thread"
    if vm_t > max(ioctl_t, futex_t, wait_t, sleep_t) or vm_c > max(ioctl_c, futex_c, wait_c, 1):
        return "likely allocation/VM-heavy thread"
    if sleep_t > 0 and sleep_t >= max(ioctl_t, futex_t, wait_t, vm_t):
        return "likely timer/sleep-dominated thread"
    return "mixed/helper thread"

def merge_stats(per_file):
    merged_counts = defaultdict(int)
    merged_time = defaultdict(float)
    merged_cat_counts = defaultdict(int)
    merged_cat_time = defaultdict(float)

    for s in per_file:
        for k, v in s["syscall_counts"].items():
            merged_counts[k] += v
        for k, v in s["syscall_time"].items():
            merged_time[k] += v
        for k, v in s["category_counts"].items():
            merged_cat_counts[k] += v
        for k, v in s["category_time"].items():
            merged_cat_time[k] += v

    return {
        "syscall_counts": dict(merged_counts),
        "syscall_time": dict(merged_time),
        "category_counts": dict(merged_cat_counts),
        "category_time": dict(merged_cat_time),
    }

def fmt_time(x: float) -> str:
    return f"{x:.6f}s"

def print_focus_block(label, counts, times):
    print(f"{label}")
    for name in FOCUS_SYSCALLS:
        c = counts.get(name, 0)
        t = times.get(name, 0.0)
        if c > 0 or t > 0:
            print(f"  {name:16s} count={c:8d}  time={fmt_time(t)}")

def print_top5(label, counts, times):
    print(label)
    top_by_time = sorted(times.items(), key=lambda kv: kv[1], reverse=True)[:5]
    for name, t in top_by_time:
        print(f"  {name:16s} time={fmt_time(t)}  count={counts.get(name, 0)}")

def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <strace_folder>", file=sys.stderr)
        sys.exit(1)

    folder = sys.argv[1]
    files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and (f.startswith("trace") or f.startswith("strace"))
    )

    if not files:
        print("No trace files found. Expected files like trace.* or strace.*", file=sys.stderr)
        sys.exit(1)

    per_file = [summarise_file(p) for p in files]
    merged = merge_stats(per_file)

    run_name = os.path.basename(os.path.abspath(folder))
    print(f"RUN: {run_name}")
    print("=" * (5 + len(run_name)))

    for s in per_file:
        print(f"\nFILE: {os.path.basename(s['path'])}")
        print(f"  parsed_lines={s['parsed_lines']} total_lines={s['total_lines']} unfinished={s['unfinished']} resumed={s['resumed']}")
        print(f"  role_guess={role_guess(s)}")
        print_focus_block("  focus syscalls:", s["syscall_counts"], s["syscall_time"])
        print_top5("  top by time:", s["syscall_counts"], s["syscall_time"])

    print(f"\nCOMBINED TOTALS: {run_name}")
    print_focus_block("  focus syscalls:", merged["syscall_counts"], merged["syscall_time"])

    print("\n  category totals:")
    for cat in sorted(merged["category_counts"].keys()):
        print(
            f"    {cat:12s} count={merged['category_counts'].get(cat, 0):8d}  "
            f"time={fmt_time(merged['category_time'].get(cat, 0.0))}"
        )

    print_top5("\n  top syscalls by time:", merged["syscall_counts"], merged["syscall_time"])

if __name__ == "__main__":
    main()
