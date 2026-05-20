# Phase 3 Remaining Roadmap: May 20 to June 12

**Day 0 = Wednesday 20 May**  
**Final target = Friday 12 June**  
**Report writing begins = Friday 29 May**  
**From June 1 onward: report work happens every day.**

## Core Objective

Deliver a minimal, well-evaluated Phase 3 prototype:

> A phase-scoped GPU work-unit abstraction with LC/BE-aware policy, evaluated through microbenchmarks, event logs, strace/eBPF, and supported by limited real-workload validation.

The priority is not to build every possible extension. The priority is to produce one strong central result, then use eBPF, LD_PRELOAD, vLLM, and optional scheduling extensions to strengthen the report.

---

# Day 0–3: Policy Lock-In  
**May 20–23**

## Goal

Turn the working shared-memory state into a real admission-control policy mechanism.

The current experimental direction is evidence-led: earlier runs suggest that chunked BE work causes limited LC P95/P99 degradation even with multiple producers, while overlapping long BE phases are the main source of LC tail-latency degradation. Therefore, the initial policy should target **BE-long admission**, not BE-chunk throttling by default.

## Tasks

- Clean up phase accounting:
  - active LC count
  - active BE-long count
  - active BE-short/chunk count
  - final counters return to zero
- Add structured policy snapshots:
  - pid/tid
  - timestamp
  - event type: phase begin, phase end, policy decision, final state
  - phase type
  - workload class
  - granularity
  - active LC/BE state
  - BE-long policy counters
  - BE-short/chunk observation counters
- Add configurable BE-long admission control:
  - environment variable or CLI argument for the long-BE concurrency cap, e.g. `BE_LONG_LIMIT`
  - environment variable or CLI argument for retry/sleep delay, e.g. `BE_DELAY_US`
- Implement first policy:
  - if an incoming `BE_LONG` phase observes `active_be_long >= BE_LONG_LIMIT`, delay admission and re-check
  - initial setting: `BE_LONG_LIMIT=1`, meaning only one BE-long phase may be active at a time
  - apply this before the BE-long phase is pushed onto the active phase stack and before `active_be_long` is incremented
  - do not throttle chunked BE by default unless later experiments show it is necessary
  - do not pretend long BE can be interrupted mid-kernel; the control point is admission before the long phase begins
- Run small sanity tests:
  - LC alone
  - BE long alone
  - BE chunked alone
  - LC + BE long + BE chunked
  - LC + multiple BE-long producers with `BE_LONG_LIMIT=1`

## Counter Definitions

- `active_lc`: current number of active latency-critical phases. Incremented after an LC phase is admitted/begins. Decremented when that LC phase ends. Must return to zero at process/system shutdown.
- `active_be_long`: current number of admitted BE-long phases. Incremented only after the BE-long admission policy allows the phase to begin. Decremented when that BE-long phase ends. This is the main state variable used by the initial policy.
- `active_be_short`: current number of active BE-short/chunk phases. Incremented when a chunked BE phase begins and decremented when it ends. Used to observe coexistence and validate that chunked BE is not being unnecessarily throttled.
- `policy_checks_total`: number of phase-begin events that reached the policy layer. This confirms that phase admission is actually routed through the policy mechanism.
- `be_long_checks`: number of BE-long phase-begin events checked by the policy.
- `be_long_admitted_immediate`: number of BE-long phases admitted without delay because `active_be_long < BE_LONG_LIMIT` at the point of admission.
- `be_long_delayed_admissions`: number of BE-long admission attempts that were delayed at least once because `active_be_long >= BE_LONG_LIMIT`.
- `be_long_delay_loops`: total number of delay/retry iterations applied to BE-long work. This may be greater than `be_long_delayed_admissions` if one phase sleeps and re-checks multiple times before admission.
- `be_long_total_delay_us`: cumulative delay applied to BE-long admission, measured in microseconds. This is the main BE policy cost metric.
- `be_short_checks`: number of BE-short/chunk phase-begin events observed by the policy layer.
- `be_short_admitted_immediate`: number of BE-short/chunk phases admitted without delay. In the initial policy this should normally equal `be_short_checks`.
- `be_short_delayed_admissions`: number of BE-short/chunk phases delayed by policy. For the initial policy this should remain zero; if non-zero, the implementation is no longer a pure BE-long admission policy.
- `final_counter_mismatch`: diagnostic counter or final assertion result indicating that at least one active counter did not return to zero. This should be zero/false for every successful run.

## Exit Criteria

- Three-way run completes reliably.
- LC, BE-long, and BE-short/chunk phases coexist in shared state.
- `BE_LONG_LIMIT` and/or `BE_DELAY_US` visibly changes BE-long behaviour.
- BE-short/chunk work is observed but not throttled by default.
- Final active counters return to zero.

---

# Day 4–6: Core Policy Evaluation  
**May 24–26**

## Goal

Produce the central result: LC latency versus BE throughput under BE-long admission control.

## Tasks

Run baseline cases:

- LC alone
- BE long alone
- BE chunked alone
- LC + BE long, no policy
- LC + BE chunked, no policy
- LC + mixed BE long/chunked, no policy

Run BE-long policy sweep:

- `BE_LONG_LIMIT=1`, `BE_DELAY_US=50us`
- `BE_LONG_LIMIT=1`, `BE_DELAY_US=100us`
- `BE_LONG_LIMIT=1`, `BE_DELAY_US=250us`
- `BE_LONG_LIMIT=1`, `BE_DELAY_US=500us`
- `BE_LONG_LIMIT=1`, `BE_DELAY_US=1000us`
- `BE_LONG_LIMIT=1`, `BE_DELAY_US=2000us`
- optional: compare `BE_LONG_LIMIT=2` if 4 BE-long producers are used

Measure:

- LC request mean/p50/p95/p99
- LC prefill/sync/decode p95/p99
- BE-long throughput
- BE-chunked throughput
- BE-long active time
- BE-chunked active time
- BE-long delayed admission count
- BE-long delay-loop count
- BE-long total admission delay
- BE-short/chunk delayed admission count, expected to remain zero in the initial policy
- LC/BE overlap

## Exit Criteria

- Main policy plot exists:
  - x-axis: BE-long admission-control setting, e.g. delay or long-BE concurrency cap
  - left y-axis: LC p95/p99
  - right y-axis: BE-long throughput
- Results show whether limiting overlapping BE-long phases improves LC tails and what it costs BE.
- Results preserve the distinction that chunked BE is comparatively well-behaved and does not require default throttling.

---

# Day 7–8: Evaluation Depth  
**May 27–28**

## Goal

Show the policy result is not a one-off.

## Tasks

Run a small controlled matrix:

- BE mode:
  - long
  - chunked
- BE workers:
  - 1
  - 4
- Delay settings:
  - `0us`
  - `500us`
  - `2000us`

Add kernel-size variation only if time permits:

- small
- medium
- large

Analyse:

- where BE-long admission control works
- where BE-long admission control fails
- whether chunked BE remains safe enough to avoid throttling
- whether more BE-long workers increase LC tail pressure
- whether the long-BE concurrency cap gives a cleaner trade-off than delaying all BE work

## Exit Criteria

- Clear long-vs-chunked comparison.
- Clear 1-BE vs 4-BE comparison.
- Clear evidence that the policy targets the workload form that actually harms LC latency.
- At least one failure/limitation case documented.

---

# Day 9: Start Report + eBPF Setup  
**May 29**

## Goal

Start the report earlier, while beginning eBPF evaluation.

## Report Tasks

Start rough versions of:

- Introduction
- Background
- Phase 2 motivation
- Problem statement
- High-level abstraction argument

Do not aim for polish yet. The goal is to create the report skeleton and begin turning the project into a written argument.

## Engineering Tasks

Begin eBPF/bpftrace setup for:

- context switches
- scheduler switches
- off-CPU time
- wakeups if feasible

## Exit Criteria

- Report skeleton exists.
- Core argument is written in rough form.
- eBPF script has started, even if not yet final.

---

# Day 10–11: eBPF Evaluation + Report Drafting  
**May 30–31**

## Goal

Use eBPF as the main host-side evaluation extension.

## Engineering Tasks

Collect eBPF data for selected runs:

- LC alone
- LC + BE no policy
- LC + BE with policy

Keep this focused. eBPF is an evaluation metric, not a policy mechanism.

## Report Tasks

Continue writing:

- Background
- Phase 2 motivation
- Abstraction design
- Prototype overview

## Exit Criteria

- eBPF data exists for at least 2–3 important runs.
- One eBPF summary table or plot exists.
- Report has enough structure to continue daily from June 1.

---

# Day 12–14: Report Every Day + Main Results Cleanup  
**June 1–3**

## Goal

Make report writing a daily activity while stabilising the main results.

## Daily Report Tasks

Write or revise every day:

- Introduction
- Background
- Motivation
- Abstraction design
- Prototype architecture
- Methodology

## Engineering Tasks

Clean:

- event-log parser
- policy parser
- result tables
- graph labels
- graph captions

Re-run any core experiment that looks suspicious.

## Exit Criteria

- Main policy result is in the report as a draft figure/table.
- Evaluation methodology is written.
- Plots are readable enough to discuss.

---

# Day 15–17: LD_PRELOAD Extension + Report  
**June 4–6**

## Goal

Attempt LD_PRELOAD as a true extension while continuing report writing every day.

## Engineering Tasks

Minimum target:

- intercept `cudaStreamSynchronize`
- emit phase begin/end around sync
- run on a simple CUDA program without source modification

Stretch target:

- intercept `cudaLaunchKernel`
- emit submission phase
- combine launch + sync traces

Frame correctly:

- explicit API = precise semantics
- LD_PRELOAD = lower-friction integration but weaker semantic precision

## Daily Report Tasks

Write:

- implementation section
- policy design section
- evaluation methodology
- first results section

## Exit Criteria

- Either a working LD_PRELOAD demo exists, or a clear partial-result/limitation note exists.
- Report continues progressing every day.
- Do not let LD_PRELOAD block the main report.

---

# Day 18–19: vLLM Validation + Report  
**June 7–8**

## Goal

Use vLLM as a short relevance validation, not as a second full evaluation.

## Engineering/Analysis Tasks

Summarise existing vLLM experiments:

- short prompts
- long prompts
- concurrency
- staggered/simultaneous runs

Extract evidence for:

- submission-like pressure
- completion/scheduler-like pressure
- host-visible syscall/scheduler behaviour

Connect to Phase 3:

- real serving stacks also expose host-visible pressure
- request/phase boundaries would help attribute this pressure
- microbenchmark remains the primary controlled evaluation

## Daily Report Tasks

Write:

- main microbenchmark results
- eBPF evaluation section
- vLLM validation section
- LD_PRELOAD extension section if available

## Exit Criteria

- vLLM section has 1–2 tables or plots.
- vLLM is framed as validation, not the main result.
- Report contains all major sections in draft form.

---

# Day 20: Stretch Extension Decision Point  
**June 9**

## Goal

Decide whether any final stretch extension is safe.

## Possible Extensions

### CPU affinity

- pin LC and BE to same/different CPU cores
- evaluate whether host CPU contention changes LC tails
- useful as a scheduling-pressure extension

### CUDA stream priority

- run LC on higher-priority stream
- run BE on lower-priority stream
- compare semantic policy versus CUDA-native priority hinting

## Rule

Only attempt one of these if:

- core policy results are complete
- eBPF result is usable
- report is already in full draft form

Otherwise, write them as future work.

## Daily Report Tasks

Continue results, limitations, and discussion.

## Exit Criteria

- Clear decision: implement one stretch extension briefly, or defer both to future work.
- No disruption to final report.

---

# Day 21–23: Final Report and Submission Polish  
**June 10–12**

## Goal

Finish the report and final artefacts.

## Tasks

- Finalise plots
- Finalise captions
- Write limitations
- Write conclusion
- Add reproducibility notes
- Clean code/scripts
- Check argument consistency
- Proofread
- Fix formatting
- Prepare submission

## Exit Criteria

- Report is complete.
- Main claims are backed by figures/tables.
- Code and scripts are reproducible enough to defend.
- No new experiments unless fixing a critical hole.

---

# vLLM Timing Decision

vLLM does not need to happen before eBPF.

The primary evidence should be:

1. event logs and policy metrics
2. LC latency versus BE throughput
3. eBPF host-side evaluation

vLLM should come after that as relevance validation. It is useful, but it should not displace the core policy/eBPF evaluation. If time becomes tight, vLLM can be a compact section based on existing experiments rather than a large new experiment campaign.

---

# Priority Order

## Must Have

1. Working phase abstraction.
2. Working multiprocess shared-memory state.
3. BE pacing policy.
4. LC latency versus BE throughput evaluation.
5. Long BE versus chunked BE comparison.
6. Daily report progress from May 29 onward.

## Strong Additions

1. eBPF evaluation metric.
2. LD_PRELOAD sync wrapper.
3. vLLM validation.

## Stretch Only

1. CUDA stream priority.
2. CPU affinity.
3. Larger experiment matrix.
4. More complex LD_PRELOAD interception.

---

# Main Story to Preserve

The OS sees GPU activity through generic host mechanisms.

Phase-scoped semantics expose workload class and control boundaries.

A lightweight policy layer can use those semantics to pace BE work under LC activity.

The benefit depends on granularity: chunked BE appears comparatively well-behaved in the current measurements, while overlapping long BE phases create the main LC tail-latency degradation and must be controlled at admission because they cannot be interrupted mid-kernel.

eBPF strengthens the host-side evaluation.

LD_PRELOAD demonstrates a lower-friction integration path.

vLLM validates that similar host-visible pressure appears in real serving workloads.

