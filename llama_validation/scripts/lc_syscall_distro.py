#!/usr/bin/env python3
"""
lc_syscall_distributions.py  (replicate-aware)

FYP Ch 7 §eBPF -- LC-side syscall-latency decomposition.

Single-replicate mode (original behaviour):
    python lc_syscall_distributions.py \
        --matrix-root <dir-containing-caseX_*>/ \
        --out-dir     ./lc_analysis \
        --cases       C D J K

Multi-replicate mode (new):
    python lc_syscall_distributions.py \
        --replicates-root <dir-containing-repNN/caseX_*/> \
        --out-dir         ./lc_analysis \
        --cases           C D E F J K

If you point --matrix-root at a dir that itself contains repNN/ subdirs the
script auto-detects that and treats it as multi-rep. So you can just pass
the OUT_ROOT_BASE the matrix runner produced and it does the right thing.

Per-rep methodology
-------------------
For each replicate, every event is bucketed by (kind, return-cohort,
thread role) and a tail summary (p50/p95/p99/p999/max) is computed *within*
that replicate. Across-rep aggregation then operates on the resulting
per-rep summaries -- mean and stdev of p99 across reps, count of reps in
which the cap-side p99 was numerically lower than the none-side p99, etc.

This is correct because the 10 replicates are independent draws from the
same workload distribution. Pooling raw events across reps before
percentiling would give a tighter point estimate but destroys the
replicate structure and you can no longer say anything about run-to-run
variability.

Outputs (all CSVs in --out-dir)
-------------------------------
  lc_syscall_distributions.csv
      Tidy long-format. One row per (rep, case, kind, cohort, thread_role).

  lc_syscall_compare_per_rep.csv
      For each rep, the C/D, G/H, J/K, E/F none-vs-cap comparison.
      One row per (rep, comparison, kind, cohort, thread_role).
      Read this when you want to see the spread of individual deltas.

  lc_syscall_compare_summary.csv
      Across-rep aggregation. One row per (comparison, kind, cohort,
      thread_role). The headline file. Columns include:
        n_reps                       (10 if all reps had data for this row)
        mean_none_p99_us, std_..._p99_us
        mean_cap_p99_us,  std_cap_p99_us
        mean_p99_delta_pct, std_p99_delta_pct
        n_reps_cap_below_none_p99    (count of reps where cap_p99 < none_p99)
        (same set repeated for p95)

The n_reps_cap_below_none_p99 column is the most directly interpretable
signal: with 10 reps under a null hypothesis (no effect), this count is
Binomial(10, 0.5), so ~5/10 is exactly what no-effect looks like, 8/10 is
borderline (p ~ 0.055 one-sided), 9/10 is suggestive (p ~ 0.011), 10/10 is
strong (p ~ 0.001).

Cohort split (unchanged)
    poll/ppoll/epoll_wait/epoll_pwait : woken(ret>0) / timeout(ret==0) / error(ret<0)
    futex/futex_waitv                  : woken(ret==0) / timeout(-110) / eagain(-11) / other
    ioctl                              : all

Thread role split (unchanged)
    cuda-EvtHandlr vs app
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# ---- Case metadata -----------------------------------------------------------

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

TARGET_KINDS = {
    'poll', 'ppoll', 'epoll_wait', 'epoll_pwait',
    'futex', 'futex_waitv',
    'ioctl',
}

COMPARE_PAIRS = [
    ('be_first sequential (C/D)',  'C', 'D'),
    ('lc_first long-LC (G/H)',     'G', 'H'),
    ('lc_cont continuous (J/K)',   'J', 'K'),
    ('be_short neg-control (E/F)', 'E', 'F'),
]

PID_REGEX = re.compile(r"Process\s*\[?\s*(\d+)")

# ---- PID extraction ----------------------------------------------------------

def extract_pid(stdout_log_path):
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

def cohort_for(kind, ret):
    if kind in ('poll', 'ppoll', 'epoll_wait', 'epoll_pwait'):
        if ret > 0:  return 'woken'
        if ret == 0: return 'timeout'
        return 'error'
    if kind in ('futex', 'futex_waitv'):
        if ret == 0:    return 'woken'
        if ret == -110: return 'timeout'
        if ret == -11:  return 'eagain'
        return 'other'
    return 'all'

# ---- Percentiles (stdlib only) -----------------------------------------------

def percentile(sorted_values, p):
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

def parse_case(case_letter, case_dir, lc_pid, be_pid):
    jsonl = case_dir / 'ebpf_events.jsonl'
    if not jsonl.exists():
        print(f"    [warn] no ebpf_events.jsonl in {case_dir}", file=sys.stderr)
        return []

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
                continue
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

    print(f"    case {case_letter}: events total={n_total}  "
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

# ---- Replicate discovery ----------------------------------------------------

def discover_replicates(args):
    """
    Return list of (rep_label, root_path) tuples.

    --replicates-root  : require repNN/ subdirs inside.
    --matrix-root      : auto-detect: if repNN/ subdirs present, treat as
                         multi-rep; else single matrix root with rep='single'.
    """
    if args.replicates_root:
        root = Path(args.replicates_root)
        rep_dirs = sorted(
            [d for d in root.iterdir()
             if d.is_dir() and re.match(r'^rep\d+$', d.name)],
            key=lambda d: d.name)
        if not rep_dirs:
            raise SystemExit(f"[error] no repNN/ subdirs in {root}")
        return [(d.name, d) for d in rep_dirs]

    if not args.matrix_root:
        raise SystemExit("[error] must specify --matrix-root or --replicates-root")

    root = Path(args.matrix_root)
    rep_dirs = sorted(
        [d for d in root.iterdir()
         if d.is_dir() and re.match(r'^rep\d+$', d.name)],
        key=lambda d: d.name)
    if rep_dirs:
        return [(d.name, d) for d in rep_dirs]
    return [('single', root)]

# ---- Per-rep parsing --------------------------------------------------------

def parse_replicate(rep_label, root, cases):
    print(f"\n>>> replicate: {rep_label}  ({root})", file=sys.stderr)
    rows = []
    for letter in cases:
        meta = CASE_META.get(letter)
        if meta is None:
            print(f"  [skip] unknown case letter {letter!r}", file=sys.stderr)
            continue
        subdir = root / meta[0]
        if not subdir.exists():
            print(f"  [skip] missing case dir {subdir}", file=sys.stderr)
            continue

        lc_pid = extract_pid(subdir / 'lc_events' / 'server_stdout.log')
        be_pid = extract_pid(subdir / 'be_events' / 'server_stdout.log')
        print(f"  case {letter}: LC pid={lc_pid}  BE pid={be_pid}",
              file=sys.stderr)

        if lc_pid is None:
            print(f"    [skip] no LC pid resolvable", file=sys.stderr)
            continue

        case_rows = parse_case(letter, subdir, lc_pid, be_pid)
        for r in case_rows:
            r['rep'] = rep_label
        rows.extend(case_rows)
    return rows

# ---- Comparison rows --------------------------------------------------------

def build_comparison_rows_for_rep(rep_rows):
    idx = defaultdict(dict)
    for r in rep_rows:
        key = (r['kind'], r['cohort'], r['thread_role'])
        idx[key][r['case']] = r

    metrics = ['n', 'p50_us', 'p95_us', 'p99_us', 'p999_us', 'max_us']
    delta_metrics = [('p50', 'p50_us'), ('p95', 'p95_us'),
                     ('p99', 'p99_us'), ('p999', 'p999_us')]

    def delta(a, b):
        if a is None or b is None:
            return (None, None)
        d = b - a
        return (d, (d / a * 100.0) if a else None)

    out = []
    for label, none_case, cap_case in COMPARE_PAIRS:
        for key, by_case in sorted(idx.items()):
            if none_case not in by_case or cap_case not in by_case:
                continue
            kind, cohort, thread_role = key
            n_row = by_case[none_case]
            c_row = by_case[cap_case]
            row = {
                'comparison': label,
                'kind': kind,
                'cohort': cohort,
                'thread_role': thread_role,
            }
            for m in metrics:
                row[f'none_{m}'] = n_row[m]
                row[f'cap_{m}']  = c_row[m]
            for short, full in delta_metrics:
                d, dp = delta(n_row[full], c_row[full])
                row[f'{short}_delta_us']  = d
                row[f'{short}_delta_pct'] = dp
            out.append(row)
    return out

# ---- Summary aggregation ----------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None

def _stdev(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) >= 2 else None

def build_summary(per_rep_comparison_rows):
    grouped = defaultdict(list)
    for r in per_rep_comparison_rows:
        key = (r['comparison'], r['kind'], r['cohort'], r['thread_role'])
        grouped[key].append(r)

    out = []
    for key, rep_rows in grouped.items():
        none_p99 = [r['none_p99_us'] for r in rep_rows]
        cap_p99  = [r['cap_p99_us']  for r in rep_rows]
        none_p95 = [r['none_p95_us'] for r in rep_rows]
        cap_p95  = [r['cap_p95_us']  for r in rep_rows]
        d99_pct  = [r['p99_delta_pct'] for r in rep_rows]
        d95_pct  = [r['p95_delta_pct'] for r in rep_rows]

        n_below_99 = sum(
            1 for r in rep_rows
            if r['cap_p99_us'] is not None and r['none_p99_us'] is not None
            and r['cap_p99_us'] < r['none_p99_us'])
        n_below_95 = sum(
            1 for r in rep_rows
            if r['cap_p95_us'] is not None and r['none_p95_us'] is not None
            and r['cap_p95_us'] < r['none_p95_us'])

        out.append({
            'comparison': key[0],
            'kind':       key[1],
            'cohort':     key[2],
            'thread_role': key[3],
            'n_reps':     len(rep_rows),
            'mean_none_p99_us': _mean(none_p99),
            'std_none_p99_us':  _stdev(none_p99),
            'mean_cap_p99_us':  _mean(cap_p99),
            'std_cap_p99_us':   _stdev(cap_p99),
            'mean_p99_delta_pct': _mean(d99_pct),
            'std_p99_delta_pct':  _stdev(d99_pct),
            'n_reps_cap_below_none_p99': n_below_99,
            'mean_none_p95_us': _mean(none_p95),
            'std_none_p95_us':  _stdev(none_p95),
            'mean_cap_p95_us':  _mean(cap_p95),
            'std_cap_p95_us':   _stdev(cap_p95),
            'mean_p95_delta_pct': _mean(d95_pct),
            'std_p95_delta_pct':  _stdev(d95_pct),
            'n_reps_cap_below_none_p95': n_below_95,
        })
    return out

# ---- CSV writers ------------------------------------------------------------

def write_csv(rows, path, fieldnames=None):
    if not rows:
        print(f"[warn] no rows to write to {path}", file=sys.stderr)
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}  ({len(rows)} rows)", file=sys.stderr)

# ---- Main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--matrix-root',
                     help='Single matrix dir, or dir containing repNN/ subdirs.')
    src.add_argument('--replicates-root',
                     help='Dir containing repNN/ subdirs (explicit).')
    ap.add_argument('--out-dir', default='./lc_analysis')
    ap.add_argument('--cases', nargs='+',
                    default=['C', 'D', 'E', 'F', 'G', 'H', 'J', 'K'])
    args = ap.parse_args()

    reps = discover_replicates(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Discovered {len(reps)} replicate(s): "
          f"{[r[0] for r in reps]}", file=sys.stderr)

    # ---- Parse all replicates ----
    all_rows = []
    for rep_label, root in reps:
        all_rows.extend(parse_replicate(rep_label, root, args.cases))

    if not all_rows:
        print("\n[error] no rows produced -- check PID regex and dir layout",
              file=sys.stderr)
        sys.exit(1)

    # ---- Tidy CSV with rep column first ----
    tidy_fieldnames = ['rep'] + [k for k in all_rows[0].keys() if k != 'rep']
    write_csv(all_rows, out / 'lc_syscall_distributions.csv',
              fieldnames=tidy_fieldnames)

    # ---- Per-rep comparison ----
    per_rep_rows = []
    by_rep = defaultdict(list)
    for r in all_rows:
        by_rep[r['rep']].append(r)
    for rep_label, rep_rows in by_rep.items():
        cmp_rows = build_comparison_rows_for_rep(rep_rows)
        for c in cmp_rows:
            c['rep'] = rep_label
        per_rep_rows.extend(cmp_rows)

    if per_rep_rows:
        cmp_fieldnames = ['rep'] + [k for k in per_rep_rows[0].keys() if k != 'rep']
        write_csv(per_rep_rows,
                  out / 'lc_syscall_compare_per_rep.csv',
                  fieldnames=cmp_fieldnames)

    # ---- Summary across reps ----
    summary_rows = build_summary(per_rep_rows)
    summary_rows.sort(key=lambda r: (r['comparison'], r['kind'],
                                     r['cohort'], r['thread_role']))
    write_csv(summary_rows, out / 'lc_syscall_compare_summary.csv')

    # ---- Stdout headline: app-thread woken cohort, the load-bearing rows ----
    print("\n=== HEADLINE: app-thread woken-cohort p99 across reps ===",
          file=sys.stderr)
    hdr = (f"{'comparison':<32} {'kind':<10} {'cohort':<8} "
           f"{'n_reps':>6} {'mean_none_p99':>14} {'mean_cap_p99':>14} "
           f"{'mean_d%':>8} {'cap<none':>10}")
    print(hdr, file=sys.stderr)
    print("-" * len(hdr), file=sys.stderr)
    for r in summary_rows:
        if r['thread_role'] != 'app':
            continue
        if r['cohort'] not in ('woken', 'all'):
            continue
        mn = r['mean_none_p99_us']
        mc = r['mean_cap_p99_us']
        md = r['mean_p99_delta_pct']
        if mn is None or mc is None:
            continue
        print(f"{r['comparison']:<32} {r['kind']:<10} {r['cohort']:<8} "
              f"{r['n_reps']:>6} {mn:>14.1f} {mc:>14.1f} "
              f"{(md if md is not None else 0):>7.2f}% "
              f"{r['n_reps_cap_below_none_p99']:>4}/{r['n_reps']:<4}",
              file=sys.stderr)

if __name__ == '__main__':
    main()
