#!/usr/bin/env python3
"""
lc_syscall_analysis_v2.py

LC-side eBPF syscall analysis for the llama.cpp policy matrix.

Purpose
-------
For each selected matrix case, parse eBPF JSONL syscall-duration events,
filter to the LC llama-server process, and compute:

  - syscall count
  - total syscall time
  - syscall count per LC request
  - syscall time per LC request
  - p50 / p95 / p99 / p99.9 / max / mean duration

It also produces none-vs-cap comparison CSVs.

This script is designed for your current setup where PIDs are recovered from:
    <case>/lc_events/server_stdout.log
    <case>/be_events/server_stdout.log

but it will prefer:
    <case>/pids.txt

if you add that later.

Outputs
-------
<out-dir>/lc_syscall_audit.csv
    Checks whether PID/event resolution looks sane.

<out-dir>/lc_syscall_distributions.csv
    Long-format per-case LC syscall distributions.

<out-dir>/lc_syscall_compare_none_vs_cap.csv
    Wide none-vs-cap comparison per regime.

<out-dir>/lc_syscall_interesting_changes.csv
    Ranked rows likely worth inspecting first.

Usage
-----
python3 lc_syscall_analysis_v2.py \
  --matrix-root /home/sc3321/FYP/llama_validation/runs/llama_phase_matrix_YYYYMMDD_HHMMSS \
  --out-dir ./lc_analysis \
  --cases C D E F G H J K
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Case metadata
# -----------------------------------------------------------------------------

CASE_META = {
    'A': ('caseA_lc_alone_none',           'lc_alone',      'none', 'control'),
    'B': ('caseB_be_long_alone_none',      'be_alone',      'none', 'control'),
    'C': ('caseC_lc_be_long_none',         'be_first',      'none', 'sequential'),
    'D': ('caseD_lc_be_long_policy',       'be_first',      'cap',  'sequential'),
    'E': ('caseE_lc_be_short_none',        'be_short',      'none', 'neg_control'),
    'F': ('caseF_lc_be_short_policy',      'be_short',      'cap',  'neg_control'),
    'G': ('caseG_lc_first_be_long_none',   'lc_first',      'none', 'long_lc'),
    'H': ('caseH_lc_first_be_long_policy', 'lc_first',      'cap',  'long_lc'),
    'I': ('caseI_lc_cont_alone_none',      'lc_cont_alone', 'none', 'control'),
    'J': ('caseJ_lc_cont_be_long_none',    'lc_cont',       'none', 'continuous'),
    'K': ('caseK_lc_cont_be_long_policy',  'lc_cont',       'cap',  'continuous'),
}

COMPARE_PAIRS = [
    ('be_first sequential C/D',  'C', 'D'),
    ('be_short neg-control E/F', 'E', 'F'),
    ('lc_first long-LC G/H',     'G', 'H'),
    ('lc_cont continuous J/K',   'J', 'K'),
]

# Include sleep syscalls as sanity, but do not automatically headline them for LC.
TARGET_KINDS = {
    'ioctl',
    'poll', 'ppoll',
    'epoll_wait', 'epoll_pwait',
    'futex', 'futex_waitv',
    'nanosleep', 'clock_nanosleep',
}

EVENT_FILE_CANDIDATES = [
    'ebpf_events.jsonl',
    'syscall_events.jsonl',
    'ioctl_events.jsonl',
]

# Current runner did not write pids.txt, but this supports it if added later.
PIDS_TXT_CANDIDATES = [
    'pids.txt',
    'pid.txt',
]

# Tolerant server stdout PID regexes.
PID_REGEXES = [
    re.compile(r"Process\s*\[?\s*(\d+)"),
    re.compile(r"\bPID\s*[:=]\s*(\d+)"),
    re.compile(r"\bpid\s*[:=]\s*(\d+)"),
]


# -----------------------------------------------------------------------------
# PID and file resolution
# -----------------------------------------------------------------------------

def read_pid_from_pids_txt(case_dir: Path, key: str) -> Optional[int]:
    """
    Try case_dir/pids.txt style files.

    Expected lines:
        LC_PID=12345
        BE_PID=12346
    """
    for name in PIDS_TXT_CANDIDATES:
        path = case_dir / name
        if not path.exists():
            continue

        try:
            for line in path.read_text(errors='replace').splitlines():
                line = line.strip()
                if not line or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                if k.strip() == key:
                    v = v.strip()
                    return int(v) if v else None
        except Exception:
            continue

    return None


def read_pid_from_server_stdout(stdout_path: Path) -> Optional[int]:
    """
    Extract PID from server stdout.

    This is intentionally tolerant because llama-server / runner output formats
    are not always stable.
    """
    if not stdout_path.exists():
        return None

    try:
        with open(stdout_path, 'r', errors='replace') as f:
            for line in f:
                for rx in PID_REGEXES:
                    m = rx.search(line)
                    if m:
                        return int(m.group(1))
    except Exception:
        return None

    return None


def resolve_lc_be_pids(case_dir: Path) -> Tuple[Optional[int], Optional[int], str, str]:
    """
    Return (lc_pid, be_pid, lc_pid_source, be_pid_source).
    """
    lc = read_pid_from_pids_txt(case_dir, 'LC_PID')
    be = read_pid_from_pids_txt(case_dir, 'BE_PID')

    lc_source = 'pids.txt' if lc is not None else ''
    be_source = 'pids.txt' if be is not None else ''

    if lc is None:
        p = case_dir / 'lc_events' / 'server_stdout.log'
        lc = read_pid_from_server_stdout(p)
        lc_source = str(p) if lc is not None else 'missing'

    if be is None:
        p = case_dir / 'be_events' / 'server_stdout.log'
        be = read_pid_from_server_stdout(p)
        be_source = str(p) if be is not None else 'missing'

    return lc, be, lc_source, be_source


def resolve_event_file(case_dir: Path) -> Optional[Path]:
    for name in EVENT_FILE_CANDIDATES:
        p = case_dir / name
        if p.exists():
            return p
    return None


def count_jsonl_lines(path: Path) -> Optional[int]:
    if not path.exists():
        return None

    n = 0
    try:
        with open(path, 'r', errors='replace') as f:
            for line in f:
                if line.strip():
                    n += 1
    except Exception:
        return None

    return n


def resolve_lc_request_count(case_dir: Path) -> Optional[int]:
    """
    Count measured LC client requests. This excludes warmup if the runner wrote
    warmup separately, which is what we want for per-request normalization.
    """
    return count_jsonl_lines(case_dir / 'lc_client.jsonl')


# -----------------------------------------------------------------------------
# Classification and statistics
# -----------------------------------------------------------------------------

def cohort_for(kind: str, ret: int) -> str:
    """
    Split return cohorts so timeout ceilings do not contaminate woken waits.
    """
    if kind in ('poll', 'ppoll', 'epoll_wait', 'epoll_pwait'):
        if ret > 0:
            return 'woken'
        if ret == 0:
            return 'timeout'
        return 'error'

    if kind in ('futex', 'futex_waitv'):
        if ret == 0:
            return 'woken'
        if ret == -110:
            return 'timeout'   # ETIMEDOUT
        if ret == -11:
            return 'eagain'    # EAGAIN / EWOULDBLOCK
        return 'other'

    if kind in ('nanosleep', 'clock_nanosleep'):
        if ret == 0:
            return 'completed'
        if ret == -4:
            return 'interrupted'  # EINTR
        return 'other'

    return 'all'


def thread_role_from_comm(comm: str) -> str:
    """
    CUDA event handler threads have a very different role from request/app
    threads. Keep them apart.
    """
    comm = (comm or '').strip()

    if comm == 'cuda-EvtHandlr':
        return 'cuda-EvtHandlr'

    # Keep room for common helper naming if it appears.
    if 'cuda' in comm.lower():
        return 'cuda-other'

    return 'app'


def percentile(sorted_values: List[float], p: float) -> Optional[float]:
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return sorted_values[0]

    k = (n - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def summarise(values: List[float], lc_requests: Optional[int]) -> Dict[str, Optional[float]]:
    n = len(values)

    if n == 0:
        return {
            'n': 0,
            'total_us': None,
            'total_ms': None,
            'count_per_lc_request': None,
            'total_ms_per_lc_request': None,
            'p50_us': None,
            'p95_us': None,
            'p99_us': None,
            'p999_us': None,
            'max_us': None,
            'mean_us': None,
        }

    s = sorted(values)
    total_us = sum(s)
    total_ms = total_us / 1000.0

    return {
        'n': n,
        'total_us': total_us,
        'total_ms': total_ms,
        'count_per_lc_request': (n / lc_requests) if lc_requests else None,
        'total_ms_per_lc_request': (total_ms / lc_requests) if lc_requests else None,
        'p50_us': percentile(s, 50),
        'p95_us': percentile(s, 95),
        'p99_us': percentile(s, 99),
        'p999_us': percentile(s, 99.9),
        'max_us': s[-1],
        'mean_us': total_us / n,
    }


def safe_float(x) -> Optional[float]:
    if x is None or x == '':
        return None
    try:
        return float(x)
    except Exception:
        return None


def delta_pair(none_val, cap_val) -> Tuple[Optional[float], Optional[float]]:
    a = safe_float(none_val)
    b = safe_float(cap_val)
    if a is None or b is None:
        return None, None
    d = b - a
    pct = (d / a * 100.0) if a != 0 else None
    return d, pct


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------

def parse_case(
    case_letter: str,
    case_dir: Path,
    lc_pid: int,
    be_pid: Optional[int],
    lc_requests: Optional[int],
    min_n_warn: int,
) -> Tuple[List[Dict], Dict]:
    event_file = resolve_event_file(case_dir)

    audit = {
        'case': case_letter,
        'case_dir': str(case_dir),
        'event_file': str(event_file) if event_file else '',
        'lc_pid': lc_pid,
        'be_pid': be_pid,
        'lc_requests': lc_requests,
        'events_total': 0,
        'events_lc': 0,
        'events_be': 0,
        'events_foreign': 0,
        'events_bad_json': 0,
        'events_missing_duration': 0,
        'events_lc_target_kind': 0,
    }

    if event_file is None:
        print(f"  [warn] no eBPF event file found in {case_dir}", file=sys.stderr)
        return [], audit

    buckets = defaultdict(list)

    with open(event_file, 'r', errors='replace') as f:
        for line in f:
            if not line.strip():
                continue

            audit['events_total'] += 1

            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                audit['events_bad_json'] += 1
                continue

            tgid = ev.get('tgid')

            if tgid == lc_pid:
                audit['events_lc'] += 1
            elif be_pid is not None and tgid == be_pid:
                audit['events_be'] += 1
                continue
            else:
                audit['events_foreign'] += 1
                continue

            kind = ev.get('kind')
            if kind not in TARGET_KINDS:
                continue

            dur_us = ev.get('dur_us')
            if dur_us is None and 'dur_ns' in ev:
                dur_us = ev['dur_ns'] / 1000.0

            if dur_us is None:
                audit['events_missing_duration'] += 1
                continue

            try:
                dur_us = float(dur_us)
            except Exception:
                audit['events_missing_duration'] += 1
                continue

            ret = int(ev.get('ret', 0))
            cohort = cohort_for(kind, ret)
            thread_role = thread_role_from_comm(ev.get('comm', ''))

            buckets[(kind, cohort, thread_role)].append(dur_us)
            audit['events_lc_target_kind'] += 1

    subdir, regime, policy, scenario = CASE_META[case_letter]

    rows = []
    for (kind, cohort, thread_role), values in sorted(buckets.items()):
        stats = summarise(values, lc_requests)
        low_n = stats['n'] < min_n_warn

        if low_n:
            print(
                f"  [warn] low n: case={case_letter} kind={kind} "
                f"cohort={cohort} thread_role={thread_role} n={stats['n']}",
                file=sys.stderr,
            )

        rows.append({
            'case': case_letter,
            'case_dir_name': subdir,
            'regime': regime,
            'policy': policy,
            'scenario': scenario,
            'lc_pid': lc_pid,
            'be_pid': be_pid,
            'lc_requests': lc_requests,
            'kind': kind,
            'cohort': cohort,
            'thread_role': thread_role,
            'low_n': int(low_n),
            **stats,
        })

    print(
        f"  case {case_letter}: total={audit['events_total']} "
        f"LC={audit['events_lc']} BE={audit['events_be']} "
        f"foreign={audit['events_foreign']} "
        f"LC_target={audit['events_lc_target_kind']}",
        file=sys.stderr,
    )

    return rows, audit


# -----------------------------------------------------------------------------
# Output builders
# -----------------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        print(f"[warn] no rows for {path}", file=sys.stderr)
        return

    # Stable field order: preserve first row, then append any surprise keys.
    fieldnames = list(rows[0].keys())
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {path} ({len(rows)} rows)", file=sys.stderr)


def build_comparison(all_rows: List[Dict]) -> List[Dict]:
    """
    Wide comparison for matched none/cap cases.
    """
    by_key = defaultdict(dict)

    for r in all_rows:
        key = (r['kind'], r['cohort'], r['thread_role'])
        by_key[key][r['case']] = r

    metrics = [
        'n',
        'total_ms',
        'count_per_lc_request',
        'total_ms_per_lc_request',
        'p50_us',
        'p95_us',
        'p99_us',
        'p999_us',
        'max_us',
        'mean_us',
    ]

    delta_metrics = [
        'total_ms',
        'count_per_lc_request',
        'total_ms_per_lc_request',
        'p50_us',
        'p95_us',
        'p99_us',
        'p999_us',
        'max_us',
    ]

    rows = []

    for comparison, none_case, cap_case in COMPARE_PAIRS:
        for (kind, cohort, thread_role), cases in sorted(by_key.items()):
            if none_case not in cases or cap_case not in cases:
                continue

            nrow = cases[none_case]
            crow = cases[cap_case]

            row = {
                'comparison': comparison,
                'none_case': none_case,
                'cap_case': cap_case,
                'regime': nrow['regime'],
                'scenario': nrow['scenario'],
                'kind': kind,
                'cohort': cohort,
                'thread_role': thread_role,
                'none_low_n': nrow.get('low_n'),
                'cap_low_n': crow.get('low_n'),
                'none_lc_requests': nrow.get('lc_requests'),
                'cap_lc_requests': crow.get('lc_requests'),
            }

            for m in metrics:
                row[f'none_{m}'] = nrow.get(m)
                row[f'cap_{m}'] = crow.get(m)

            for m in delta_metrics:
                d, pct = delta_pair(nrow.get(m), crow.get(m))
                row[f'{m}_delta'] = d
                row[f'{m}_delta_pct'] = pct

            rows.append(row)

    return rows


def build_interesting_changes(compare_rows: List[Dict]) -> List[Dict]:
    """
    Smaller ranked view. This is not a statistical test; it is a triage file.
    """
    ranked = []

    for r in compare_rows:
        # App-side first. CUDA event handler rows can be inspected later.
        if r.get('thread_role') != 'app':
            continue

        none_n = safe_float(r.get('none_n')) or 0
        cap_n = safe_float(r.get('cap_n')) or 0
        if min(none_n, cap_n) < 20:
            continue

        total_req_delta_pct = safe_float(r.get('total_ms_per_lc_request_delta_pct'))
        p95_delta_pct = safe_float(r.get('p95_us_delta_pct'))
        p99_delta_pct = safe_float(r.get('p99_us_delta_pct'))

        # Ranking score: prefer total activity/request movement, then p95/p99.
        score_parts = [
            abs(total_req_delta_pct) if total_req_delta_pct is not None else 0.0,
            abs(p95_delta_pct) if p95_delta_pct is not None else 0.0,
            abs(p99_delta_pct) if p99_delta_pct is not None else 0.0,
        ]
        score = max(score_parts)

        out = dict(r)
        out['triage_score'] = score
        ranked.append(out)

    ranked.sort(key=lambda x: safe_float(x.get('triage_score')) or 0.0, reverse=True)
    return ranked


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrix-root', required=True,
                    help='Path to llama_phase_matrix_<timestamp>/')
    ap.add_argument('--out-dir', default='./lc_analysis')
    ap.add_argument('--cases', nargs='+',
                    default=['C', 'D', 'E', 'F', 'G', 'H', 'J', 'K'])
    ap.add_argument('--min-n-warn', type=int, default=20,
                    help='Warn/flag rows with fewer than this many samples.')
    args = ap.parse_args()

    root = Path(args.matrix_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    audit_rows = []

    for letter in args.cases:
        letter = letter.upper().strip()
        if letter not in CASE_META:
            print(f"[skip] unknown case letter {letter!r}", file=sys.stderr)
            continue

        subdir_name, regime, policy, scenario = CASE_META[letter]
        case_dir = root / subdir_name

        if not case_dir.exists():
            print(f"[skip] missing case dir: {case_dir}", file=sys.stderr)
            continue

        lc_pid, be_pid, lc_src, be_src = resolve_lc_be_pids(case_dir)
        lc_requests = resolve_lc_request_count(case_dir)

        print(
            f"case {letter}: LC pid={lc_pid} ({lc_src}) "
            f"BE pid={be_pid} ({be_src}) LC_requests={lc_requests}",
            file=sys.stderr,
        )

        if lc_pid is None:
            audit_rows.append({
                'case': letter,
                'case_dir': str(case_dir),
                'event_file': '',
                'lc_pid': '',
                'be_pid': be_pid,
                'lc_pid_source': lc_src,
                'be_pid_source': be_src,
                'lc_requests': lc_requests,
                'events_total': 0,
                'events_lc': 0,
                'events_be': 0,
                'events_foreign': 0,
                'events_bad_json': 0,
                'events_missing_duration': 0,
                'events_lc_target_kind': 0,
                'status': 'skip_no_lc_pid',
            })
            print(f"  [skip] no LC PID resolvable for case {letter}", file=sys.stderr)
            continue

        rows, audit = parse_case(
            case_letter=letter,
            case_dir=case_dir,
            lc_pid=lc_pid,
            be_pid=be_pid,
            lc_requests=lc_requests,
            min_n_warn=args.min_n_warn,
        )

        audit['lc_pid_source'] = lc_src
        audit['be_pid_source'] = be_src
        audit['status'] = 'ok'

        audit_rows.append(audit)
        all_rows.extend(rows)

    write_csv(out / 'lc_syscall_audit.csv', audit_rows)

    if not all_rows:
        print("[error] no LC distribution rows produced.", file=sys.stderr)
        print("Check PID extraction, event filename, and case directory layout.", file=sys.stderr)
        sys.exit(1)

    write_csv(out / 'lc_syscall_distributions.csv', all_rows)

    comparison_rows = build_comparison(all_rows)
    write_csv(out / 'lc_syscall_compare_none_vs_cap.csv', comparison_rows)

    interesting_rows = build_interesting_changes(comparison_rows)
    write_csv(out / 'lc_syscall_interesting_changes.csv', interesting_rows)

    print("\nNext inspect:", file=sys.stderr)
    print(f"  {out / 'lc_syscall_audit.csv'}", file=sys.stderr)
    print(f"  {out / 'lc_syscall_interesting_changes.csv'}", file=sys.stderr)


if __name__ == '__main__':
    main()
