#!/usr/bin/env python3
"""
lc_syscall_distributions.py

FYP Ch 7 §eBPF — LC-side syscall-latency decomposition.

For each case directory under the matrix root, this script:

  1. Resolves the LC server PID and BE server PID from
        <case>/lc_events/server_stdout.log
        <case>/be_events/server_stdout.log
     (these are written by the patched matrix runner; both processes share
      comm="llama-server", so external labelling is the only non-circular
      way to disambiguate).

  2. Streams <case>/ebpf_events.jsonl and accumulates per-LC-syscall durations,
     grouped by (syscall kind, return-cohort, thread role). Cohort splits are
     applied because waiting syscalls have ceiling/error returns that, if
     pooled with the "real" wait population, dominate the p99 and produce a
     metric that mostly reflects timeout rate rather than contention:

        poll, ppoll, epoll_wait, epoll_pwait
            woken    ret > 0    -- woken by FD readiness; real wait time
            timeout  ret == 0   -- structural ceiling (e.g. 100 ms timeout)
            error    ret < 0

        futex, futex_waitv
            woken    ret == 0
            timeout  ret == -110 (ETIMEDOUT)
            eagain   ret == -11
            other

        ioctl
            all      (no explicit timeout semantics on this path; this is
                      the syscall where driver-side blocking is visible)

     Thread role is split into `cuda-EvtHandlr` vs `app` so that the CUDA
     event-handler's structurally bounded polling does not contaminate the
     request-handling tail.

  3. Writes two CSVs into --out-dir:

        lc_syscall_distributions.csv
            tidy long-format: one row per (case, kind, cohort, thread_role)
            with n, p50, p95, p99, p999, max, mean (all µs).

        lc_syscall_compare_none_vs_cap.csv
            pivoted comparison view: for each (none, cap) case pair within
            an ordering regime, per (kind, cohort, thread_role), reports
            none-side and cap-side stats and the p50/p95/p99/p999 deltas
            in absolute µs and percent.

Assumptions worth verifying before trusting the output:
  * One eBPF trace per case (no replicates). If replicates exist, the
    comparison view needs a U-test or bootstrap on per-run p99s — not
    implemented here.
  * PID line in server_stdout.log matches the regex r"Process\s*\[?\s*(\d+)".
    If the actual format differs, the script will print "LC pid=None" for
    that case and skip it; adjust PID_REGEX below.

Usage:
    python lc_syscall_distributions.py \
        --matrix-root /home/sc3321/FYP/llama_validation/runs/llama_phase_matrix_20260604_213624 \
        --out-dir     ./lc_analysis \
        --cases       C D G H J K
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---- Case metadata -----------------------------------------------------------

# letter -> (subdir name, ordering regime, policy mode, scenario class)
CASE_META = {
    'A': ('caseA_lc_alone_none',          'lc_alone',     'none', 'control'),
    'B': ('caseB_be_long_alone_none',     'be_alone',     'none', 'control'),
    'C': ('caseC_lc_be_long_none',        'be_first',     'none', 'sequential'),
    'D': ('caseD_lc_be_long_policy',      'be_first',     'cap',  'sequential'),
    'E': ('caseE_lc_be_short_none',       'be_short',     'none', 'neg_control'),
    'F': ('caseF_lc_be_short_policy',     'be_short',     'cap',  'neg_control'),
    'G': ('caseG_lc_first_be_long_none',  'lc_first',     'none', 'long_lc'),
    'H': ('caseH_lc_first_be_long_policy','lc_first',     'cap',  'long_lc'),
    'I': ('caseI_lc_cont_alone_none',     'lc_cont_alone','none', 'control'),
    'J': ('caseJ_lc_cont_be_long_none',   'lc_cont',      'none', 'continuous'),
    'K': ('caseK_lc_cont_be_long_policy', 'lc_cont',      'cap',  'continuous'),
}

# Syscall kinds analysed on LC-side
TARGET_KINDS = {
    'poll', 'ppoll', 'epoll_wait', 'epoll_pwait',
    'futex', 'futex_waitv',
    'ioctl',
}

# Comparison pairs: (label, none-case, cap-case)
COMPARE_PAIRS = [
    ('be_first sequential (C/D)',  'C', 'D'),
    ('lc_first long-LC (G/H)',     'G', 'H'),
    ('lc_cont continuous (J/K)',   'J', 'K'),
    ('be_short neg-control (E/F)', 'E', 'F'),
]

# ---- PID extraction ----------------------------------------------------------

# Tolerant: matches "Process[12345]", "Process [12345]", "Process 12345", etc.
# If your server_stdout.log uses a different prefix, change this.
PID_REGEX = re.compile(r"Process\s*\[?\s*(\d+)")

def extract_pid(stdout_log_path: Path):
    """Return int PID from the first matching line, or None."""
    try:
        with open(stdout_log_path, 'r', errors='replace') as f:
            for line in f:
                m = PID_REGEX.search(line)
                if m:
                    return int(m.group(1))
    except FileNotFoundError:
        return None
    return None

# ---- Cohort assignment -------------------------------------------------------

def cohort_for(kind: str, ret: int) -> str:
    if kind in ('poll', 'ppoll', 'epoll_wait', 'epoll_pwait'):
        if ret > 0:  return 'woken'
        if ret == 0: return 'timeout'
        return 'error'
    if kind in ('futex', 'futex_waitv'):
        if ret == 0:    return 'woken'
        if ret == -110: return 'timeout'   # ETIMEDOUT
        if ret == -11:  return 'eagain'    # EAGAIN / EWOULDBLOCK
        return 'other'
    # ioctl + anything else: no cohort split
    return 'all'

# ---- Percentiles (stdlib only) -----------------------------------------------

def percentile(sorted_values, p: float):
    """Linear-interpolation percentile. p in [0, 100]. Returns None if empty."""
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return sorted_values[0]
    k = (n - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])

def summarise(values):
    """n, p50, p95, p99, p999, max, mean (µs) from a list of durations."""
    n = len(values)
    if n == 0:
        return {'n': 0, 'p50_us': None, 'p95_us': None, 'p99_us': None,
                'p999_us': None, 'max_us': None, 'mean_us': None}
    s = sorted(values)
    return {
        'n':       n,
        'p50_us':  percentile(s, 50),
        'p95_us':  percentile(s, 95),
        'p99_us':  percentile(s, 99),
        'p999_us': percentile(s, 99.9),
        'max_us':  s[-1],
        'mean_us': sum(s) / n,
    }

# ---- Per-case parse ----------------------------------------------------------

def parse_case(case_letter: str, case_dir: Path, lc_pid: int, be_pid):
    """Stream the case's JSONL, return list of summary rows for the LC process."""
    jsonl = case_dir / 'ebpf_events.jsonl'
    if not jsonl.exists():
        print(f"  [warn] no ebpf_events.jsonl in {case_dir}", file=sys.stderr)
        return []

    # buckets[(kind, cohort, thread_role)] = [dur_us, ...]
    buckets = defaultdict(list)
    n_total = n_lc = n_be = n_foreign = 0

    with open(jsonl, 'r') as f:
        for line in f:
            n_total += 1
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue

            tgid = ev.get('tgid')
            if tgid == lc_pid:
                n_lc += 1
            elif be_pid is not None and tgid == be_pid:
                n_be += 1
                continue   # BE not analysed in this script
            else:
                n_foreign += 1
                continue

            kind = ev.get('kind')
            if kind not in TARGET_KINDS:
                continue

            ret = ev.get('ret', 0)
            cohort = cohort_for(kind, ret)
            comm = (ev.get('comm') or '').strip()
            thread_role = 'cuda-EvtHandlr' if comm == 'cuda-EvtHandlr' else 'app'

            dur_us = ev.get('dur_us')
            if dur_us is None and 'dur_ns' in ev:
                dur_us = ev['dur_ns'] / 1000.0
            if dur_us is None:
                continue

            buckets[(kind, cohort, thread_role)].append(dur_us)

    print(f"  case {case_letter}: events total={n_total}  "
          f"lc={n_lc}  be={n_be}  foreign={n_foreign}", file=sys.stderr)

    _, regime, policy, scenario = (case_dir.name, *CASE_META[case_letter][1:])
    rows = []
    for (kind, cohort, thread_role), values in sorted(buckets.items()):
        stats = summarise(values)
        rows.append({
            'case': case_letter,
            'regime': regime,
            'policy': policy,
            'scenario': scenario,
            'kind': kind,
            'cohort': cohort,
            'thread_role': thread_role,
            **stats,
        })
    return rows

# ---- Comparison view ---------------------------------------------------------

def build_comparison(all_rows, out_path: Path):
    """Wide CSV: for each compare pair, side-by-side none vs cap with deltas."""
    idx = defaultdict(dict)
    for r in all_rows:
        key = (r['kind'], r['cohort'], r['thread_role'])
        idx[key][r['case']] = r

    metrics = ['n', 'p50_us', 'p95_us', 'p99_us', 'p999_us', 'max_us']
    delta_pcts = [('p50', 'p50_us'), ('p95', 'p95_us'),
                  ('p99', 'p99_us'), ('p999', 'p999_us')]

    fieldnames = ['comparison', 'kind', 'cohort', 'thread_role']
    for m in metrics:
        fieldnames += [f'none_{m}', f'cap_{m}']
    for short, _ in delta_pcts:
        fieldnames += [f'{short}_delta_us', f'{short}_delta_pct']

    def delta(a, b):
        if a is None or b is None:
            return (None, None)
        d = b - a
        return (d, (d / a * 100.0) if a else None)

    with open(out_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for label, none_case, cap_case in COMPARE_PAIRS:
            for key, by_case in sorted(idx.items()):
                if none_case not in by_case or cap_case not in by_case:
                    continue
                kind, cohort, thread_role = key
                n_row = by_case[none_case]
                c_row = by_case[cap_case]
                row = {'comparison': label, 'kind': kind,
                       'cohort': cohort, 'thread_role': thread_role}
                for m in metrics:
                    row[f'none_{m}'] = n_row[m]
                    row[f'cap_{m}']  = c_row[m]
                for short, full in delta_pcts:
                    d, dp = delta(n_row[full], c_row[full])
                    row[f'{short}_delta_us']  = d
                    row[f'{short}_delta_pct'] = dp
                w.writerow(row)

# ---- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrix-root', required=True,
                    help='Path to llama_phase_matrix_<timestamp>/')
    ap.add_argument('--out-dir', default='./lc_analysis')
    ap.add_argument('--cases', nargs='+',
                    default=['C', 'D', 'G', 'H', 'J', 'K'],
                    help='Case letters to process (default: C D G H J K)')
    args = ap.parse_args()

    root = Path(args.matrix_root)
    out  = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for letter in args.cases:
        meta = CASE_META.get(letter)
        if meta is None:
            print(f"[skip] unknown case letter {letter!r}", file=sys.stderr)
            continue
        subdir = root / meta[0]
        if not subdir.exists():
            print(f"[skip] missing case dir {subdir}", file=sys.stderr)
            continue

        lc_pid = extract_pid(subdir / 'lc_events' / 'server_stdout.log')
        be_pid = extract_pid(subdir / 'be_events' / 'server_stdout.log')
        print(f"case {letter}: LC pid={lc_pid}  BE pid={be_pid}", file=sys.stderr)

        if lc_pid is None:
            print(f"  [skip] no LC pid resolvable for case {letter}",
                  file=sys.stderr)
            continue

        all_rows.extend(parse_case(letter, subdir, lc_pid, be_pid))

    if not all_rows:
        print("[error] no rows produced — check PID regex and dir layout",
              file=sys.stderr)
        sys.exit(1)

    tidy_path = out / 'lc_syscall_distributions.csv'
    with open(tidy_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"wrote {tidy_path}  ({len(all_rows)} rows)", file=sys.stderr)

    cmp_path = out / 'lc_syscall_compare_none_vs_cap.csv'
    build_comparison(all_rows, cmp_path)
    print(f"wrote {cmp_path}", file=sys.stderr)

if __name__ == '__main__':
    main()
