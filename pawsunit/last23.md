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

Turn the working shared-memory state into a real policy mechanism.

## Tasks

- Clean up phase accounting:
  - active LC count
  - active BE-long count
  - active BE-short/chunk count
  - final counters return to zero
- Add structured policy snapshots:
  - pid/tid
  - event type
  - phase type
  - workload class
  - granularity
  - active LC/BE state
  - wait/admit/throttle counters
- Add configurable BE delay:
  - environment variable or CLI argument
  - e.g. `BE_DELAY_US`
- Implement first policy:
  - if LC is active, delay BE chunk admission
  - apply mainly at chunked BE boundaries
  - do not pretend long BE can be interrupted mid-kernel
- Run small sanity tests:
  - LC alone
  - BE long alone
  - BE chunked alone
  - LC + BE long + BE chunked

## Exit Criteria

- Three-way run completes reliably.
- LC and BE coexist in shared state.
- BE delay knob visibly changes BE behaviour.
- Final active counters return to zero.

---

# Day 4–6: Core Policy Evaluation  
**May 24–26**

## Goal

Produce the central result: LC latency versus BE throughput under policy.

## Tasks

Run baseline cases:

- LC alone
- BE long alone
- BE chunked alone
- LC + BE long, no policy
- LC + BE chunked, no policy

Run delay sweep:

- `0us`
- `50us`
- `100us`
- `250us`
- `500us`
- `1000us`
- `2000us`
- `5000us`

Measure:

- LC request mean/p50/p95/p99
- LC prefill/sync/decode p95/p99
- BE throughput
- BE active time
- BE wait count
- BE total wait time
- LC/BE overlap

## Exit Criteria

- Main policy plot exists:
  - x-axis: BE delay
  - left y-axis: LC p95/p99
  - right y-axis: BE throughput
- Results show whether policy improves LC tails and what it costs BE.

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

- where policy works
- where policy fails
- whether chunked BE is more controllable than long BE
- whether more BE workers increase LC tail pressure

## Exit Criteria

- Clear long-vs-chunked comparison.
- Clear 1-BE vs 4-BE comparison.
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

The benefit depends on granularity: chunked BE exposes safe intervention points, while long BE does not.

eBPF strengthens the host-side evaluation.

LD_PRELOAD demonstrates a lower-friction integration path.

vLLM validates that similar host-visible pressure appears in real serving workloads.

