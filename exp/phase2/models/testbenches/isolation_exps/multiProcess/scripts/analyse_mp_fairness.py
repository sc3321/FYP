#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Generic strace line with absolute timestamp and syscall duration.
# Works best with: strace -ff -ttt -T ...
SYSCALL_RE = re.compile(
    r"""
    ^(?P<ts>\d+\.\d+)\s+
    (?P<name>[a-zA-Z_][a-zA-Z0-9_]*)
    \(.*\)
    .*?
    <(?P<dur>\d+\.\d+)>
    \s*$
    """,
    re.VERBOSE,
)

# Capture write(...) lines so we can recover marker payload text.
WRITE_RE = re.compile(
    r"""
    ^(?P<ts>\d+\.\d+)\s+
    write
    \(
    (?P<fd>\d+),
    \s*
    (?P<payload>".*?"|'.*?'|[^,]+)
    ,
    .*?
    \)
    .*?
    <(?P<dur>\d+\.\d+)>
    \s*$
    """,
    re.VERBOSE,
)

# Permissive marker detection.
# It will recognize things like:
#   ITER_START 17
#   MARKER: ITER_START 17
#   iteration_start=17
#   iter 17 start
#
# And same for END.
START_PATTERNS = [
    re.compile(r"ITER(?:ATION)?[_\s-]*START[^0-9]*([0-9]+)", re.IGNORECASE),
    re.compile(r"START[^0-9]*ITER(?:ATION)?[^0-9]*([0-9]+)", re.IGNORECASE),
    re.compile(r"ITER[^0-9]*([0-9]+)[^A-Za-z]*START", re.IGNORECASE),
]

END_PATTERNS = [
    re.compile(r"ITER(?:ATION)?[_\s-]*END[^0-9]*([0-9]+)", re.IGNORECASE),
    re.compile(r"END[^0-9]*ITER(?:ATION)?[^0-9]*([0-9]+)", re.IGNORECASE),
    re.compile(r"ITER[^0-9]*([0-9]+)[^A-Za-z]*END", re.IGNORECASE),
]

WAIT_NAMES = {"poll", "ppoll", "epoll_wait", "futex"}
DRIVER_NAMES = {"ioctl"}


@dataclass
class Event:
    ts: float
    name: str
    dur: float
    raw: str


@dataclass
class Marker:
    ts: float
    kind: str   # "start" | "end"
    iteration: int
    raw_payload: str
    file_name: str


@dataclass
class IterationWindow:
    iteration: int
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    source_file: Optional[str] = None
    wait_time: float = 0.0
    ioctl_time: float = 0.0
    wait_count: int = 0
    ioctl_count: int = 0

    @property
    def duration(self) -> Optional[float]:
        if self.start_ts is None or self.end_ts is None:
            return None
        return self.end_ts - self.start_ts

    @property
    def complete(self) -> bool:
        return self.start_ts is not None and self.end_ts is not None


@dataclass
class ThreadTrace:
    file_name: str
    events: List[Event] = field(default_factory=list)
    markers: List[Marker] = field(default_factory=list)


@dataclass
class RunTrace:
    run_dir: Path
    threads: List[ThreadTrace]

    def all_events(self) -> List[Event]:
        out = []
        for t in self.threads:
            out.extend(t.events)
        out.sort(key=lambda e: e.ts)
        return out

    def all_markers(self) -> List[Marker]:
        out = []
        for t in self.threads:
            out.extend(t.markers)
        out.sort(key=lambda m: m.ts)
        return out


def clean_payload(payload: str) -> str:
    payload = payload.strip()
    if len(payload) >= 2 and ((payload[0] == '"' and payload[-1] == '"') or (payload[0] == "'" and payload[-1] == "'")):
        payload = payload[1:-1]
    payload = payload.encode("utf-8", "replace").decode("unicode_escape", "replace")
    return payload


def detect_marker(payload: str) -> Optional[Tuple[str, int]]:
    for pat in START_PATTERNS:
        m = pat.search(payload)
        if m:
            return ("start", int(m.group(1)))
    for pat in END_PATTERNS:
        m = pat.search(payload)
        if m:
            return ("end", int(m.group(1)))
    return None


def parse_trace_file(path: Path) -> ThreadTrace:
    trace = ThreadTrace(file_name=path.name)

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.rstrip("\n")

            m = SYSCALL_RE.match(s)
            if m:
                try:
                    trace.events.append(
                        Event(
                            ts=float(m.group("ts")),
                            name=m.group("name"),
                            dur=float(m.group("dur")),
                            raw=s,
                        )
                    )
                except Exception:
                    pass

            wm = WRITE_RE.match(s)
            if wm:
                try:
                    ts = float(wm.group("ts"))
                    payload = clean_payload(wm.group("payload"))
                    marker = detect_marker(payload)
                    if marker is not None:
                        kind, iteration = marker
                        trace.markers.append(
                            Marker(
                                ts=ts,
                                kind=kind,
                                iteration=iteration,
                                raw_payload=payload,
                                file_name=path.name,
                            )
                        )
                except Exception:
                    pass

    trace.events.sort(key=lambda e: e.ts)
    trace.markers.sort(key=lambda m: m.ts)
    return trace


def load_run(run_dir: Path) -> RunTrace:
    files = sorted(
        p for p in run_dir.iterdir()
        if p.is_file() and (p.name.startswith("strace.") or p.name.startswith("trace."))
    )
    return RunTrace(run_dir=run_dir, threads=[parse_trace_file(p) for p in files])


def build_iteration_windows(run: RunTrace) -> Dict[int, IterationWindow]:
    """
    Build per-iteration windows from markers, then attribute wait/ioctl events whose
    timestamps fall inside each window.
    """
    windows: Dict[int, IterationWindow] = {}

    # Use markers first
    for marker in run.all_markers():
        w = windows.setdefault(marker.iteration, IterationWindow(iteration=marker.iteration))
        if marker.kind == "start":
            if w.start_ts is None or marker.ts < w.start_ts:
                w.start_ts = marker.ts
                w.source_file = marker.file_name
        elif marker.kind == "end":
            if w.end_ts is None or marker.ts > w.end_ts:
                w.end_ts = marker.ts

    # Attribute events into windows
    complete_windows = [w for w in windows.values() if w.complete]
    complete_windows.sort(key=lambda w: w.start_ts if w.start_ts is not None else math.inf)

    events = run.all_events()
    wi = 0
    n = len(complete_windows)

    for e in events:
        while wi < n and complete_windows[wi].end_ts is not None and e.ts > complete_windows[wi].end_ts:
            wi += 1
        if wi >= n:
            break
        w = complete_windows[wi]
        if w.start_ts is None or w.end_ts is None:
            continue
        if w.start_ts <= e.ts <= w.end_ts:
            if e.name in WAIT_NAMES:
                w.wait_time += e.dur
                w.wait_count += 1
            if e.name in DRIVER_NAMES:
                w.ioctl_time += e.dur
                w.ioctl_count += 1

    return windows


def fmt_s(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{x:.6f}s"


def percentile(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def summarize_run(run: RunTrace, windows: Dict[int, IterationWindow], limit: int = 20) -> None:
    markers = run.all_markers()
    complete = [w for w in windows.values() if w.complete and w.duration is not None]
    complete.sort(key=lambda w: w.iteration)

    print(f"RUN: {run.run_dir}")
    print("=" * (len(str(run.run_dir)) + 5))
    print()
    print(f"Threads parsed: {len(run.threads)}")
    print(f"Markers found : {len(markers)}")
    print(f"Iterations with complete START/END pairs: {len(complete)}")
    print()

    durations = [w.duration for w in complete if w.duration is not None]
    waits = [w.wait_time for w in complete]
    ioctls = [w.ioctl_time for w in complete]

    if durations:
        print("Iteration duration stats:")
        print(f"  min   {fmt_s(min(durations))}")
        print(f"  p50   {fmt_s(percentile(durations, 0.50))}")
        print(f"  p95   {fmt_s(percentile(durations, 0.95))}")
        print(f"  p99   {fmt_s(percentile(durations, 0.99))}")
        print(f"  max   {fmt_s(max(durations))}")
        print()

    if waits:
        print("Per-iteration wait-time stats (poll/futex/etc attributed inside iteration window):")
        print(f"  min   {fmt_s(min(waits))}")
        print(f"  p50   {fmt_s(percentile(waits, 0.50))}")
        print(f"  p95   {fmt_s(percentile(waits, 0.95))}")
        print(f"  p99   {fmt_s(percentile(waits, 0.99))}")
        print(f"  max   {fmt_s(max(waits))}")
        print()

    if ioctls:
        print("Per-iteration ioctl-time stats:")
        print(f"  min   {fmt_s(min(ioctls))}")
        print(f"  p50   {fmt_s(percentile(ioctls, 0.50))}")
        print(f"  p95   {fmt_s(percentile(ioctls, 0.95))}")
        print(f"  p99   {fmt_s(percentile(ioctls, 0.99))}")
        print(f"  max   {fmt_s(max(ioctls))}")
        print()

    print("First iterations:")
    for w in complete[:limit]:
        print(
            f"  iter={w.iteration:>5} "
            f"dur={fmt_s(w.duration):>12} "
            f"wait={fmt_s(w.wait_time):>12} "
            f"ioctl={fmt_s(w.ioctl_time):>12} "
            f"wait_ct={w.wait_count:>4} "
            f"ioctl_ct={w.ioctl_count:>4}"
        )
    print()

    if len(complete) > limit:
        print("Last iterations:")
        for w in complete[-limit:]:
            print(
                f"  iter={w.iteration:>5} "
                f"dur={fmt_s(w.duration):>12} "
                f"wait={fmt_s(w.wait_time):>12} "
                f"ioctl={fmt_s(w.ioctl_time):>12} "
                f"wait_ct={w.wait_count:>4} "
                f"ioctl_ct={w.ioctl_count:>4}"
            )
        print()

    worst = sorted(complete, key=lambda w: w.duration if w.duration is not None else -1.0, reverse=True)[:10]
    print("Worst iterations by duration:")
    for w in worst:
        print(
            f"  iter={w.iteration:>5} "
            f"dur={fmt_s(w.duration):>12} "
            f"wait={fmt_s(w.wait_time):>12} "
            f"ioctl={fmt_s(w.ioctl_time):>12} "
            f"wait_ct={w.wait_count:>4} "
            f"ioctl_ct={w.ioctl_count:>4}"
        )
    print()


def compare_runs(
    a_name: str,
    a_windows: Dict[int, IterationWindow],
    b_name: str,
    b_windows: Dict[int, IterationWindow],
    limit: int = 30,
) -> None:
    common_iters = sorted(set(a_windows.keys()) & set(b_windows.keys()))
    common_complete = [
        i for i in common_iters
        if a_windows[i].complete and b_windows[i].complete
        and a_windows[i].duration is not None and b_windows[i].duration is not None
    ]

    print("PAIR COMPARISON")
    print("===============")
    print()
    print(f"A = {a_name}")
    print(f"B = {b_name}")
    print(f"Comparable complete iterations: {len(common_complete)}")
    print()

    if not common_complete:
        print("No complete overlapping iterations found.")
        return

    rows = []
    for i in common_complete:
        aw = a_windows[i]
        bw = b_windows[i]
        ad = aw.duration or 0.0
        bd = bw.duration or 0.0
        rows.append(
            (
                i,
                ad,
                bd,
                ad - bd,
                aw.wait_time,
                bw.wait_time,
                aw.wait_time - bw.wait_time,
                aw.ioctl_time,
                bw.ioctl_time,
                aw.ioctl_time - bw.ioctl_time,
            )
        )

    duration_deltas = [r[3] for r in rows]
    abs_duration_deltas = [abs(x) for x in duration_deltas]

    print("Cross-process iteration duration skew (A - B):")
    print(f"  mean abs skew: {fmt_s(sum(abs_duration_deltas) / len(abs_duration_deltas))}")
    print(f"  p50 abs skew : {fmt_s(percentile(abs_duration_deltas, 0.50))}")
    print(f"  p95 abs skew : {fmt_s(percentile(abs_duration_deltas, 0.95))}")
    print(f"  max abs skew : {fmt_s(max(abs_duration_deltas))}")
    print()

    print("Most asymmetric iterations by duration:")
    for r in sorted(rows, key=lambda x: abs(x[3]), reverse=True)[:limit]:
        iter_id, ad, bd, dd, awt, bwt, wdiff, aio, bio, iodiff = r
        leader = "A slower" if dd > 0 else "B slower"
        print(
            f"  iter={iter_id:>5}  "
            f"A_dur={fmt_s(ad):>12}  "
            f"B_dur={fmt_s(bd):>12}  "
            f"delta={fmt_s(dd):>12}  "
            f"A_wait={fmt_s(awt):>12}  "
            f"B_wait={fmt_s(bwt):>12}  "
            f"{leader}"
        )
    print()

    print("Drift view (sampled every ~25 iterations):")
    step = max(1, len(rows) // 20)
    cumulative_a = 0.0
    cumulative_b = 0.0
    for idx, r in enumerate(rows):
        _, ad, bd, *_ = r
        cumulative_a += ad
        cumulative_b += bd
        if idx % step == 0 or idx == len(rows) - 1:
            print(
                f"  idx={idx:>4} "
                f"iter={r[0]:>5} "
                f"cum_A={fmt_s(cumulative_a):>12} "
                f"cum_B={fmt_s(cumulative_b):>12} "
                f"gap={fmt_s(cumulative_a - cumulative_b):>12}"
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse iteration markers in strace output to study per-iteration fairness/asymmetry."
    )
    parser.add_argument("run_a", type=Path, help="First run directory containing strace.* or trace.* files")
    parser.add_argument("run_b", type=Path, nargs="?", help="Optional second run directory")
    parser.add_argument("--limit", type=int, default=20, help="Rows to show in tables")
    args = parser.parse_args()

    run_a = load_run(args.run_a)
    win_a = build_iteration_windows(run_a)
    summarize_run(run_a, win_a, limit=args.limit)

    if args.run_b is not None:
        run_b = load_run(args.run_b)
        win_b = build_iteration_windows(run_b)
        summarize_run(run_b, win_b, limit=args.limit)
        compare_runs(str(args.run_a), win_a, str(args.run_b), win_b, limit=max(20, args.limit))


if __name__ == "__main__":
    main()
