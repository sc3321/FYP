# Revised 23-Day Phase 3 Plan  
## Evaluation-led plan for final implementation, real-workload evaluation, eBPF, LD_PRELOAD, and report delivery

**Period:** Day 0 to Day 23  
**Primary objective:** deliver a strong, defensible final-year systems project with a working OS-adjacent GPU phase abstraction, controlled evaluation, real C/C++ workload validation, host-side eBPF evidence, and a clear report narrative.

---

# 0. Core Principle

This plan is **evaluation-led**.

The project should not become:

> A clever policy library with scattered experiments.

It should become:

> A phase-scoped GPU work abstraction, evaluated first on controlled CUDA workloads and then on a real C/C++ LLM inference workload, with eBPF used to expose host-side effects and LD_PRELOAD used to demonstrate a transparent integration path.

The engineering work matters, but every engineering addition should strengthen one of these:

1. the main claim,
2. the evaluation,
3. the report argument,
4. the presentation/demo.

---

# 1. Priority Stack

## Must Have Before Report Submission

1. Working phase abstraction and shared-memory state.
2. Correct active counter accounting:
   - `active_lc`
   - `active_be_long`
   - `active_be_short`
3. BE-long admission-cap policy.
4. Controlled microbenchmark evaluation.
5. `llama.cpp` or fallback real C/C++ workload evaluation.
6. eBPF host-side evaluation for selected runs.
7. Report figures, methodology, and argument.

## Strong Additions Before Report Submission

1. Policy modes.
2. Structured policy snapshots.
3. Simple policy daemon or snapshot collector.
4. LD_PRELOAD sync wrapper.
5. Automated experiment runner and parser cleanup.

## Stretch Before Report Submission

1. Shared-memory ring buffer.
2. More complete LD_PRELOAD launch interception.
3. Larger experiment matrix.
4. More complex daemon-driven policy control.

## Presentation-Gap Polish Only

1. Better daemon output.
2. Ring-buffer polish.
3. More LD_PRELOAD coverage.
4. Cleaner demo scripts.
5. Extra plots that do not change report claims.
6. Code cleanup and README improvements.

---

# 2. Main Claim to Build Around

> Controlled microbenchmarks show that chunked BE work is comparatively controllable and causes limited LC tail degradation, while overlapping BE-long phases create stronger LC tail pressure. A phase-scoped policy can use semantic workload class and granularity to cap BE-long admission, preserving chunked BE where possible while reducing harmful long-phase overlap. This is then evaluated on a real C/C++ LLM inference workload and correlated with host-side scheduling behaviour using eBPF.

This claim is important because it avoids overclaiming.

You are **not** claiming:

> All BE work should be delayed when LC is present.

You are claiming:

> Semantic phase information lets the policy distinguish harmful BE-long overlap from comparatively benign chunked BE.

That is a much stronger systems argument.

---

# 3. Final System Shape

By the end, the project should look like this:

```text
Application / workload
    |
    | explicit phase API OR LD_PRELOAD wrapper
    v
Phase library
    |
    | updates shared state
    | emits structured events
    | applies admission policy at phaseBegin()
    v
Shared-memory policy state
    |
    | active counters
    | policy counters
    | optional snapshots/events
    v
Policy analysis layer
    |
    | parser / daemon / snapshot collector
    | experiment scripts
    v
Evaluation
    |
    | microbenchmark LC/BE results
    | llama.cpp results
    | eBPF host-side results
    v
Report
```

---

# 4. Policy Modes

Implement these modes if possible:

```text
GPU_PHASE_POLICY_MODE=none
GPU_PHASE_POLICY_MODE=naive_delay_be_when_lc
GPU_PHASE_POLICY_MODE=cap_be_long
GPU_PHASE_POLICY_MODE=cap_be_long_when_lc
```

## Mode 0: `none`

No policy action. Still update counters and emit logs.

Purpose:

```text
baseline
```

## Mode 1: `naive_delay_be_when_lc`

Delay BE admissions when LC is active.

Purpose:

```text
early/simple policy baseline
```

This may be intentionally worse or too blunt. That is fine. It gives you a prototype-improvement story.

## Mode 2: `cap_be_long`

Limit concurrent BE-long phases.

Purpose:

```text
main evidence-informed policy
```

## Mode 3: `cap_be_long_when_lc`

Limit concurrent BE-long phases only when LC is active.

Purpose:

```text
more selective policy
```

This may preserve more BE throughput while still protecting LC.

---

# 5. Canonical Counter Definitions

## Active Counters

| Counter | Definition | Increment | Decrement | Purpose |
|---|---|---|---|---|
| `active_lc` | Number of currently active latency-critical phases | After LC phase begins | At LC phase end | Tracks LC presence |
| `active_be_long` | Number of admitted BE-long phases currently active | After BE-long passes admission | At BE-long phase end | Main BE-long cap variable |
| `active_be_short` | Number of active BE-short/chunk phases | At BE-short/chunk begin | At BE-short/chunk end | Tracks chunked BE coexistence |

Important rule:

> Waiting BE-long phases must not count as active. Increment `active_be_long` only after the policy admits the phase.

---

## Policy Counters

| Counter | Definition |
|---|---|
| `policy_checks_total` | Number of phase-begin calls that reached the policy layer |
| `be_long_checks` | Number of BE-long phase-begin events checked |
| `be_long_admitted_immediate` | BE-long admissions that proceeded without delay |
| `be_long_delayed_admissions` | BE-long admissions that waited at least once |
| `be_long_delay_loops` | Total sleep/retry iterations across all BE-long admissions |
| `be_long_total_delay_us` | Total microseconds spent delaying BE-long admission |
| `be_short_checks` | Number of BE-short/chunk phase-begin events checked |
| `be_short_admitted_immediate` | BE-short/chunk admissions that proceeded without delay |
| `be_short_delayed_admissions` | BE-short/chunk admissions delayed by policy |
| `final_counter_mismatch` | Indicates leaked active counters at shutdown/final snapshot |

For the first serious policy, `be_short_delayed_admissions` should usually remain zero.

That is useful because it proves the policy is not blindly delaying all BE.

---

# 6. Day-by-Day Plan

---

# Day 0–2: Lock the Correct Policy and Counters

## Goal

Finish the core policy mechanism. Do not touch `llama.cpp` yet.

## Engineering Tasks

Implement or clean:

```text
active_lc
active_be_long
active_be_short
```

Implement policy counters:

```text
policy_checks_total
be_long_checks
be_long_admitted_immediate
be_long_delayed_admissions
be_long_delay_loops
be_long_total_delay_us
be_short_checks
be_short_admitted_immediate
be_short_delayed_admissions
final_counter_mismatch
```

Implement policy modes:

```text
POLICY_MODE=none
POLICY_MODE=naive_delay_be_when_lc
POLICY_MODE=cap_be_long
POLICY_MODE=cap_be_long_when_lc
```

Implement environment knobs:

```text
GPU_PHASE_POLICY_MODE
GPU_PHASE_BE_LONG_LIMIT
GPU_PHASE_BE_DELAY_US
GPU_PHASE_MAX_DELAY_LOOPS
GPU_PHASE_LOG_DIR
GPU_PHASE_SNAPSHOT_PATH
```

Main policy logic:

```cpp
if (incoming == BE_LONG && active_be_long >= BE_LONG_LIMIT) {
    sleep/retry until active_be_long < BE_LONG_LIMIT;
}
```

Then:

```cpp
active_be_long++;
push_phase_stack();
write_begin_event();
```

## Sanity Runs

```text
LC alone
BE-long alone
BE-chunked alone
LC + BE-long
LC + BE-chunked
LC + BE-long + BE-chunked
```

## Exit Criteria

```text
Three-way run completes.
Final counters return to zero.
BE-long cap visibly reduces overlap.
Chunked BE is observed but not unnecessarily throttled.
Policy logs show admit/delay decisions clearly.
```

---

# Day 3–5: Controlled Microbenchmark Evaluation

## Goal

Produce the first real result.

This result is the controlled foundation of the project.

## Experiments

Run:

```text
LC alone
BE-long alone
BE-chunked alone
LC + 1 BE-long, no policy
LC + 4 BE-long, no policy
LC + 1 BE-long, cap policy
LC + 4 BE-long, cap policy
LC + 4 BE-chunked, no policy
LC + 4 BE-chunked, cap policy
```

Policy settings:

```text
none
cap_BE_LONG, limit=1
cap_BE_LONG, limit=2
optional: cap_BE_LONG_when_LC_active
```

Delay settings:

```text
BE_DELAY_US=100
BE_DELAY_US=500
BE_DELAY_US=2000
```

Do not run a huge matrix. Pick enough to prove the mechanism.

## Metrics

```text
LC_REQUEST mean/p50/p95/p99
LC_PREFILL / SYNC / DECODE p95/p99 if available
BE throughput
BE active time
BE-long delayed admissions
BE-long total delay
BE-long overlap over time
final active counter values
```

## Required Plots/Tables

```text
1. LC p95/p99 by condition
2. BE throughput by condition
3. BE-long delay count / total delay by condition
4. active_be_long timeline for no policy vs cap policy
```

## Exit Criteria

You can clearly explain:

```text
What hurts LC?
What does the policy throttle?
What does it cost BE?
Why is chunked BE different from long BE?
```

This becomes the core controlled result.

---

# Day 6: Real Workload Bring-up Decision Day

## Goal

Get a real C/C++ workload running, ideally `llama.cpp`.

## Primary Target

```text
llama.cpp with CUDA backend
```

## Fallbacks

```text
Fallback 1: whisper.cpp
Fallback 2: ggml examples
Fallback 3: vLLM trace-only validation
```

## Hard Rule

Spend **one day maximum** on build/setup.

If `llama.cpp` becomes a build rabbit hole, pivot.

## Minimal Success

```text
Can run one GPU-backed inference.
Can vary prompt length.
Can vary generation length.
Can capture latency and tokens/sec.
Can run repeated requests from a script.
```

## Exit Criteria

One real workload run exists and produces measurable output.

---

# Day 7–8: Manual Phase Instrumentation in Real Workload

## Goal

Add minimal, meaningful phase scopes to the real workload.

Do not over-instrument.

## Target Semantic Phases

For `llama.cpp`, try to identify boundaries equivalent to:

```text
REQUEST
PREFILL
DECODE
TOKEN_STEP or BATCH_EVAL
```

Initial phase classes:

```text
interactive short generation = LC
long/background generation = BE_LONG
small repeated background generation = BE_SHORT / CHUNKED
```

## Engineering Tasks

Add phase begin/end around 2–3 stable locations.

Example:

```cpp
gpu_phase_begin("llama_request", LC, SHORT);
...
gpu_phase_end();
```

For background long generation:

```cpp
gpu_phase_begin("llama_background_long", BE, LONG);
...
gpu_phase_end();
```

For decode loop or repeated chunks:

```cpp
gpu_phase_begin("llama_decode_chunk", BE, SHORT);
...
gpu_phase_end();
```

## Important Constraint

Do not spend days understanding all internals.

Find stable boundaries, instrument them, and move on.

## Exit Criteria

```text
Real workload emits phase logs.
LC and BE classifications appear in shared state.
At least one LC run and one BE run are measurable.
No policy yet required.
```

---

# Day 9–10: Real Workload LC/BE Evaluation

## Goal

Produce the second major result: relevance on real inference.

## Experiments

Run:

```text
LC llama alone
BE-long llama alone
LC + BE-long, no policy
LC + BE-long, BE-long cap policy
LC + BE-short/chunked, no policy
LC + BE-short/chunked, BE-long cap policy
```

If possible:

```text
LC + 2 BE-long
LC + 4 BE-long
```

But do not overexpand.

## Workload Design

Use simple workload classes:

```text
LC:
  short prompt
  short generation
  interactive style

BE-long:
  longer prompt
  longer generation
  background batch style

BE-short:
  repeated shorter background chunks
```

## Metrics

```text
LC request latency p50/p95/p99
LC tokens/sec if meaningful
BE tokens/sec
BE total completion time
BE-long delay count
BE-long total delay
active_be_long overlap
```

## Required Output

At least one table:

| Condition | LC p50 | LC p95 | LC p99 | BE tok/s | BE delay count | BE delay total |
|---|---:|---:|---:|---:|---:|---:|

## Exit Criteria

You can say:

```text
The policy was evaluated not only on synthetic CUDA workloads but also on a real C/C++ LLM inference stack.
```

---

# Day 11: First Report Integration Pass

## Goal

Start converting results into the report.

This is not optional. Do not wait until all code is done.

## Report Sections to Draft

```text
Introduction
Problem statement
Phase 2 motivation summary
Abstraction design
Prototype architecture
Policy design
Controlled evaluation methodology
```

## Figures to Insert as Drafts

```text
Architecture diagram
Policy state diagram
Microbenchmark LC p95/p99 plot
Microbenchmark BE throughput plot
Real workload preliminary table
```

## Exit Criteria

The report contains the main story in rough form.

Even if some numbers are placeholders, the structure must exist.

---

# Day 12–13: eBPF Evaluation

## Goal

Add host-side systems evidence.

eBPF is an evaluation lens, not the main policy mechanism.

## Selected Runs Only

Do not instrument everything.

Run eBPF for:

```text
LC alone
LC + BE-long, no policy
LC + BE-long, BE-long cap policy
LC + BE-chunked, no policy
```

Use either microbenchmark runs or real workload runs. Ideally do both if cheap, but prioritise selected runs.

## Metrics

Try to capture:

```text
context switches
off-CPU time
wakeups
futex/poll/syscall timing
scheduler switches
```

## Useful Scripts

Create small scripts such as:

```text
offcpu.bt
sched_switch_summary.bt
syscall_latency.bt
run_ebpf_suite.sh
```

## Output

One compact table:

| Condition | Context switches | Off-CPU time | Futex/poll time | Notes |
|---|---:|---:|---:|---|

## Exit Criteria

You can write:

```text
eBPF confirms that the policy changes host-visible scheduling/blocking behaviour in selected cases.
```

Or, if the result is weak:

```text
eBPF shows that application-level phase effects are clearer than raw host scheduler metrics, reinforcing the need for semantic phase exposure.
```

Both are acceptable.

---

# Day 14: Parser and Plot Cleanup

## Goal

Make the results reproducible and report-ready.

## Engineering Tasks

Clean or write scripts for:

```text
event log parsing
policy snapshot parsing
LC latency summary
BE throughput summary
eBPF summary parsing
plot generation
```

## Required Outputs

```text
results_microbench.csv
results_llama.csv
results_ebpf.csv
plot_lc_tail_microbench.png/pdf
plot_be_throughput_microbench.png/pdf
plot_llama_policy.png/pdf
plot_ebpf_summary.png/pdf
```

## Exit Criteria

One command or documented sequence can regenerate the main plots.

---

# Day 15–16: LD_PRELOAD Sync Wrapper

## Goal

Demonstrate transparent integration.

This is important, but it must not derail the report.

## Minimum Target

Intercept:

```text
cudaStreamSynchronize
cudaDeviceSynchronize
```

Emit automatic phase events around sync.

Example:

```text
AUTO_SYNC_BEGIN
real cudaStreamSynchronize()
AUTO_SYNC_END
```

## Engineering Tasks

Build:

```text
libgpuphase_preload.so
```

Use:

```bash
LD_PRELOAD=./libgpuphase_preload.so ./your_cuda_program
```

## Evaluation

Run on:

```text
simple CUDA microbenchmark
optional: llama.cpp if easy
```

## Report Framing

Manual API:

```text
precise semantic classification
```

LD_PRELOAD:

```text
lower-friction integration
weaker semantic precision
```

## Exit Criteria

```text
LD_PRELOAD wrapper runs on at least one CUDA program.
Automatic sync events appear in logs.
Report can include it as an integration extension.
```

---

# Day 17: Simple Policy Daemon / Snapshot Collector

## Goal

Add OS-like architecture without making the daemon critical-path for policy decisions.

## Scope

Build a simple daemon:

```bash
phasepolicyd --snapshot-out policy_snapshots.csv
```

It should:

```text
attach to shared memory
periodically read counters
write structured snapshots
print active_lc / active_be_long / active_be_short
detect final counter mismatch if possible
```

Optional:

```text
read policy config from file
display live policy mode
```

## Do Not

Do not require every phaseBegin call to IPC into the daemon.

The policy still lives in the library for now.

## Why This Helps

It makes the design feel OS-adjacent:

```text
multiple processes publish phase state
central observer sees system-wide GPU phase activity
policy decisions become inspectable
```

## Exit Criteria

```text
Daemon runs during a microbenchmark or llama.cpp run.
Daemon emits structured CSV snapshots.
Report architecture diagram can include it.
```

---

# Day 18: Second Report Integration Pass

## Goal

Move the report from skeleton to full draft.

## Sections to Draft or Improve

```text
Implementation
Policy design
Microbenchmark evaluation
Real workload evaluation
eBPF evaluation
LD_PRELOAD extension
Limitations
```

## Required Writing

Add the prototype progression:

```text
Prototype 0: phase logging and counters
Prototype 1: naive policy
Prototype 2: BE-long cap policy
Prototype 3: real workload integration
Prototype 4: transparency/observability extensions
```

## Exit Criteria

Report is no longer a skeleton. It is a rough full draft.

---

# Day 19: Re-run Suspicious Experiments

## Goal

Stabilise the evidence.

## Tasks

Review all plots and tables.

Identify:

```text
outliers
missing baselines
runs with leaked counters
runs where BE throughput seems impossible
runs where LC p99 is dominated by one bad sample
```

Re-run only what matters.

## Do Not

Do not add a new experiment family unless the report has a critical hole.

## Exit Criteria

Main claims are backed by clean data.

---

# Day 20: Optional Ring Buffer OR Report-First Decision

## Goal

Make a hard decision.

At this point, ask:

```text
Is the report strong enough?
Are the main plots done?
Is llama.cpp evaluated?
Is eBPF evaluated?
Is LD_PRELOAD minimally working?
```

## If yes: implement a minimal shared-memory ring buffer.

## If no: skip ring buffer and write.

### Ring Buffer Scope

Only implement if it replaces/improves logging cleanly.

Minimum event:

```cpp
struct PolicyEvent {
    uint64_t ts_ns;
    uint32_t pid;
    uint32_t tid;
    uint64_t phase_id;
    uint16_t event_type;
    uint16_t phase_type;
    uint16_t workload_class;
    uint16_t granularity;
    uint32_t active_lc;
    uint32_t active_be_long;
    uint32_t active_be_short;
    uint64_t delay_us;
};
```

Minimum buffer:

```cpp
struct RingBuffer {
    std::atomic<uint64_t> head;
    std::atomic<uint64_t> dropped;
    PolicyEvent events[N];
};
```

Better but optional:

```text
sequence-numbered slots
```

## Exit Criteria

Either:

```text
Ring buffer works and daemon drains it.
```

Or:

```text
Ring buffer is deferred to presentation/future work.
```

Both are acceptable.

---

# Day 21: Final Results and Captions

## Goal

Make figures report-ready.

## Tasks

For every main figure/table, write:

```text
what it shows
why it matters
what claim it supports
what limitation it has
```

## Required Figures/Tables

Minimum:

```text
1. System architecture diagram
2. Policy state/counter diagram
3. Microbenchmark LC p95/p99 comparison
4. Microbenchmark BE throughput comparison
5. BE-long overlap or delay counter plot
6. llama.cpp real workload result table/plot
7. eBPF host-side summary table/plot
8. LD_PRELOAD integration diagram or small result
```

## Exit Criteria

No unexplained plot remains in the report.

---

# Day 22: Full Report Polish

## Goal

Turn draft into submission-quality report.

## Tasks

```text
tighten introduction
tighten problem statement
make claims conservative and precise
check all figures are referenced
check methodology is reproducible
write limitations honestly
write conclusion
proofread
format
```

## Claim Discipline

Do not claim:

```text
general GPU scheduling solution
production-ready runtime
kernel-level preemption
universal LC improvement
```

Claim:

```text
OS-adjacent semantic phase exposure
multiprocess shared state
evidence-informed BE-long admission control
controlled and real-workload evaluation
host-side eBPF observability
transparent integration path via LD_PRELOAD
```

## Exit Criteria

Report can be submitted if necessary.

---

# Day 23: Submission Lockdown

## Goal

No new technical work unless fixing a critical issue.

## Tasks

```text
final proofread
final plot export
final captions
final references
final reproducibility notes
clean repo
tag final code
backup results
prepare submission package
```

## Final Checklist

```text
Report complete.
Main claims backed by figures/tables.
Microbenchmark result included.
Real workload result included.
eBPF result included.
LD_PRELOAD extension included if working.
Limitations included.
Future work included.
Code/results reproducible enough to defend.
```

## Exit Criteria

Submit.

---

# 7. Evaluation Matrix

Keep the final matrix small.

## Controlled Microbenchmarks

| Case | Purpose |
|---|---|
| LC alone | baseline latency |
| BE-long alone | BE ideal throughput |
| BE-chunked alone | BE ideal throughput |
| LC + BE-long no policy | harmful interference |
| LC + BE-long cap policy | main policy improvement |
| LC + BE-chunked no policy | show chunked is less harmful |
| LC + BE-chunked cap policy | show policy does not unnecessarily punish chunked BE |
| LC + BE-long + BE-chunked | coexistence sanity |

## Real Workload

| Case | Purpose |
|---|---|
| LC llama alone | real workload LC baseline |
| BE-long llama alone | real BE throughput baseline |
| LC + BE-long no policy | real workload interference |
| LC + BE-long cap policy | real workload policy effect |
| optional LC + BE-short | compare chunked/short behaviour |

## eBPF

| Case | Purpose |
|---|---|
| LC alone | host baseline |
| LC + BE-long no policy | host-side interference |
| LC + BE-long cap policy | host-side policy effect |
| LC + BE-chunked no policy | compare benign/chunked case |

---

# 8. Report Structure

Suggested final report structure:

```text
1. Introduction
2. Background and Motivation
3. Phase 2 Findings / Mechanism Motivation
4. Abstraction Design
5. Prototype Implementation
   5.1 Phase API
   5.2 Shared-memory state
   5.3 Policy counters
   5.4 BE-long admission policy
   5.5 Policy daemon / snapshots
   5.6 LD_PRELOAD integration
6. Methodology
   6.1 Controlled microbenchmarks
   6.2 Real workload: llama.cpp
   6.3 eBPF host-side measurements
7. Evaluation
   7.1 Controlled LC/BE interference
   7.2 BE-long cap policy
   7.3 Real workload validation
   7.4 eBPF host-side evidence
   7.5 Overheads
8. Discussion
   8.1 Why chunked BE differs from long BE
   8.2 Limits of userspace admission control
   8.3 Manual API vs LD_PRELOAD
9. Limitations
10. Future Work
11. Conclusion
```

---

# 9. What to Push After Report Submission

There may be a 3–5 day gap between report submission and presentation. Use it carefully.

## Safe Post-Submission Improvements

These are safe because they improve the demo but do not need to support report claims.

```text
polish policy daemon output
improve ring buffer reliability
add live terminal dashboard
make plots prettier
write a better README
clean experiment runner
improve presentation diagrams
add a small demo script
extend LD_PRELOAD from sync-only to launch+sync if feasible
```

## Risky Post-Submission Work

Do not rely on the gap for:

```text
main llama.cpp evaluation
main policy result
main eBPF result
core report figures
core report argument
```

If it supports a major claim, it must exist before report submission.

---

# 10. Presentation Demo Ideas

Potential demo flow:

```text
1. Show no-policy run:
   active_be_long rises
   LC tail worsens

2. Show BE-long cap policy:
   BE-long admission delayed
   active_be_long capped
   LC tail improves or stabilises

3. Show llama.cpp run:
   real workload emits phase events

4. Show eBPF summary:
   host-side scheduling/blocking changes

5. Show LD_PRELOAD:
   CUDA sync phase events emitted without source modification
```

Optional nice visual:

```text
live daemon output:
  active_lc=1 active_be_long=1 active_be_short=3 delayed=42 total_delay_us=21000
```

---

# 11. Final Rule

From Day 11 onward, every day must include report progress.

Even if code is exciting, the report is the product that gets marked.

The implementation gives the report credibility, but the report turns the implementation into a defensible engineering contribution.

