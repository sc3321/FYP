# Phase 2 Plan --- Host-Side Visibility & Mechanism Mapping

Generated: 2026-02-13

------------------------------------------------------------------------

## Phase 2 Objective

By the end of this phase, you will:

1.  Build a reproducible host-side profiling harness.
2.  Construct a 4-vanilla calibration suite in C/C++.
3.  Profile a real serving stack (vLLM or SGLang).
4.  Map observed behavior to underlying host-visible mechanisms.
5.  Produce 3--5 defensible, mechanism-level claims that naturally lead
    into Phase 3.

Time budget: \~6--7 weeks, 10--20 hours per week.

------------------------------------------------------------------------

# Week 1 --- Toolchain Foundations & Reproducible Harness

## Goals

-   Build a repeatable profiling workflow.
-   Understand what each profiling tool actually measures.
-   Produce one complete, synchronized "profiling run."

## Work

1.  Learn and test:
    -   `strace` (syscall tracing + timing)
    -   `perf record` + flamegraph generation
    -   `perf sched` (or basic eBPF scheduling stats)
2.  Create a wrapper script that:
    -   Accepts workload parameters
    -   Runs profilers
    -   Launches workload
    -   Stops profilers
    -   Stores outputs in structured folders
3.  Define run folder format:

```{=html}
<!-- -->
```
    runs/
      run_001/
        config.json
        strace.txt
        perf.data
        flamegraph.svg
        sched.txt

## Deliverable

-   One complete profiling run on a simple GPU program.
-   Written summary: what each tool reveals and what it does NOT reveal.

## Skills Developed

-   Linux performance tooling
-   Systems reproducibility discipline
-   Interpreting flamegraphs

------------------------------------------------------------------------

# Week 2 --- 4-Vanilla Calibration Suite (C/C++ Heavy)

## Purpose

Create controlled baselines that generate distinct syscall and
scheduling signatures.

## Implement in C/C++:

1.  Compute-heavy
    -   Few long kernel launches
    -   Minimal synchronization
2.  Launch-heavy
    -   Many small kernel launches
3.  Sync-heavy
    -   Frequent host-device synchronizations
4.  Alloc-heavy
    -   Frequent allocations/deallocations or memory mapping changes

Each program must: - Have a tunable parameter - Log runtime - Be
deterministic and clean

## Deliverables

-   4 standalone C/C++ programs
-   Profiling results for each
-   Table summarizing syscall mix and CPU time distribution

## Skills Developed

-   CUDA runtime intuition
-   Driver interaction understanding
-   Mechanism fingerprinting

------------------------------------------------------------------------

# Week 3 --- Stress Testing the 4-Vanilla Suite

## Purpose

Observe scaling behavior under load and identify mechanism signatures.

## Stress Knobs

-   Concurrency (1 → 2 → 4 → 8 threads)
-   Work size
-   Sync placement

## Produce

-   Plots of:
    -   Syscall rate vs concurrency
    -   Context switches vs concurrency
    -   CPU utilization vs concurrency

## Deliverable

Short internal memo: "Under high concurrency, pattern X emerges due to
mechanism Y."

## Skills Developed

-   Scheduler intuition
-   Concurrency reasoning
-   Forming causal hypotheses

------------------------------------------------------------------------

# Week 4 --- Introduce Real Workload (Offline Inference)

## Purpose

Transition from synthetic patterns to real serving behavior.

## Steps

1.  Install and run minimal vLLM or SGLang inference script.
2.  Start with:
    -   Small model
    -   Single request
3.  Profile once using harness.

## Then sweep:

-   Context length
-   Batch size

## Deliverables

-   First real workload profile
-   Comparison to 4-vanilla signatures

## Skills Developed

-   Serving stack familiarity
-   Mapping framework behavior to OS-visible signals

------------------------------------------------------------------------

# Week 5 --- Concurrency & Serving Mode Stress

## Purpose

Push the system to high load and observe host bottlenecks.

## Stress Dimensions

-   Concurrent clients
-   QPS
-   Batching configuration

## Measure

-   Throughput
-   Latency (p50/p95)
-   Syscall mix
-   Context switch rate
-   Runqueue latency

## Deliverables

-   Latency vs concurrency plot
-   Syscall mix vs concurrency plot
-   Host bottleneck identification summary

## Skills Developed

-   Performance analysis under load
-   Throughput vs latency tradeoffs

------------------------------------------------------------------------

# Week 6 --- Mechanism Mapping & Causal Validation

## Purpose

Turn correlation into causation.

For each dominant pattern: 1. Form hypothesis 2. Modify one parameter 3.
Re-profile 4. Confirm or reject hypothesis

Example: "IOCTL rate scales with launch frequency." → Change batching →
Measure shift

## Deliverables

Mechanism Mapping Table:

  Observation   Hypothesis   Intervention   Result   Conclusion
  ------------- ------------ -------------- -------- ------------

## Skills Developed

-   Scientific rigor
-   Controlled experimentation
-   Systems-level reasoning

------------------------------------------------------------------------

# Week 7 --- Synthesis & Phase 3 Bridge

## Purpose

Translate Phase 2 findings into abstraction insight.

## Questions to Answer

-   What semantic is invisible to the OS?
-   What dominates host overhead?
-   What small interface change could improve visibility?

## Produce

-   3--5 Phase 2 Claims
-   2--3 Phase 3 Proposal Directions

## Example Phase 3 Directions

-   Phase-boundary hint export
-   Batch-structure visibility
-   Cache-pressure signaling
-   Coarse-grain dispatch grouping

## Skills Developed

-   Abstraction design
-   Bridging measurement to proposal

------------------------------------------------------------------------

# End-State of Phase 2

You will: - Understand Linux scheduling behavior under GPU load -
Interpret syscall distributions fluently - Map serving behavior to
host-visible mechanisms - Be ready to design a minimal semantic export
for Phase 3

------------------------------------------------------------------------
