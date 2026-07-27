# Phase 2 --- Host Visibility & Mechanism Mapping

Generated: 2026-02-28

------------------------------------------------------------------------

# Overview

Duration: **4.5 Weeks**

Primary Goal:

By the end of Phase 2 you will have:

1.  A structured **Mechanism Atlas** mapping CUDA runtime behaviour to
    host-visible OS mechanisms.
2.  Scheduler-level understanding of GPU completion and wake behaviour.
3.  VM subsystem understanding under GPU memory pressure.
4.  Concurrency and contention analysis under mixed workload stress.
5.  Real-world validation using a minimal AI serving stack
    (vLLM/SGLang).
6.  3--5 defensible mechanism-level claims that directly motivate Phase
    3 abstraction design.

------------------------------------------------------------------------

# What Phase 3 Will Actually Do

Phase 3 is NOT about measuring more syscalls.

Phase 3 is about designing and prototyping a **minimal abstraction
layer** that corrects the visibility and structural limitations
identified in Phase 2.

Phase 2 discovers:

-   How GPU submission is expressed as ioctl storms.
-   How GPU completion is expressed as generic futex blocking.
-   How GPU memory lifetime becomes VM churn.
-   How concurrency amplifies scheduler pressure.
-   How the OS lacks semantic grouping for GPU execution phases.

Phase 3 then:

1.  Defines a clean abstraction capturing missing semantic structure.
2.  Implements a minimal user-space prototype (C/C++) that wraps CUDA
    runtime behaviour.
3.  Demonstrates measurable improvement in:
    -   Reduced syscall amplification
    -   Reduced scheduler pressure
    -   Improved grouping of dispatch/completion events
4.  Evaluates under both microbenchmarks and real AI workloads.

Deliverable of Phase 3:

-   Clean abstraction definition (API + semantics)
-   Working prototype
-   Empirical evidence that abstraction reduces structural pressure
    identified in Phase 2

------------------------------------------------------------------------

# Timeline Overview

  Week   Focus                            Depth Level
  ------ -------------------------------- ---------------------------
  1      Structural mechanism isolation   Controlled baseline
  2      Scheduler + VM deep dive         OS subsystem reasoning
  3      Concurrency & contention         Systems interaction
  4      Mixed stress & CPU pressure      Real scheduling behaviour
  4.5    AI workload validation           Real-world confirmation

------------------------------------------------------------------------

# Week 1 --- Mechanism Atlas

## Submission Topology

Experiments: - 1000 small kernels - 100 medium kernels - 1 large/fused
kernel

Measure: - strace -c - perf record - perf sched timehist - Context
switches - CPU time in kernel

Goal: Identify dispatch amplification and driver serialization
behaviour.

Deliverable: Submission → Host mechanism mapping table.

------------------------------------------------------------------------

## Synchronisation Topology

Experiments: - Sync per iteration - Sync once - No sync

Measure: - futex counts - perf sched timehist -
/proc/`<pid>`{=html}/task/`<tid>`{=html}/wchan - Wake latency

Goal: Understand scheduler state transitions during GPU wait.

Deliverable: Precise explanation of GPU completion → scheduler
sleep/wake pattern.

------------------------------------------------------------------------

## Allocation Lifetime

Experiments: - Alloc/free per iteration - Reuse buffers - Small vs large
allocations

Measure: - mmap/munmap/mprotect - Page faults - VM region growth -
Kernel CPU time

Goal: Map GPU memory lifetime → host VM subsystem behaviour.

------------------------------------------------------------------------

## Context Lifetime

Experiments: - Cold start - Warm start - Multi-process context creation

Measure: - mmap/ioctl burst at initialization - Thread creation -
Time-to-first-kernel

Goal: Identify initialization signature and driver setup cost.

------------------------------------------------------------------------

# Week 2 --- OS Subsystem Deep Dive

## Scheduler Analysis

-   Measure runqueue delay
-   Measure wakeup latency
-   Add CPU stress (stress -c N)
-   Observe distortion of GPU completion visibility

Deliverable: Scheduler interaction memo.

------------------------------------------------------------------------

## VM Behaviour

-   Monitor VMA growth
-   Monitor page faults
-   Evaluate fragmentation under allocation stress

------------------------------------------------------------------------

## Driver Path Analysis

-   perf record -g
-   Inspect ioctl call stacks
-   Identify internal serialization

------------------------------------------------------------------------

# Week 3 --- Concurrency & Contention

## Multi-threaded Submission

-   1, 2, 4, 8 threads
-   Shared vs separate streams

Observe: - ioctl scaling - futex growth - Runqueue delay

------------------------------------------------------------------------

## Multi-process GPU Use

-   Two processes
-   With and without MPS

Observe: - Context churn - Fairness behaviour - Interference

------------------------------------------------------------------------

## Sync × Concurrency

Combine high sync with multi-thread submission.

Observe lock convoying and wake amplification.

------------------------------------------------------------------------

# Week 4 --- Mixed Pressure & CPU Contention

## Add CPU-bound workload

Run submission-heavy and sync-heavy under CPU stress.

Measure: - Wake latency - Runqueue delay - Context switches

------------------------------------------------------------------------

## Extreme Mixed Case

-   Many small kernels
-   Many threads
-   CPU contention

Identify structural limits of host-visible runtime behaviour.

------------------------------------------------------------------------

# Week 4.5 --- AI Workload Validation

Introduce minimal vLLM or SGLang setup.

Steps: 1. Small model, single client. 2. Controlled QPS sweep. 3.
Compare mechanism signatures against microbenchmarks.

Goal: Validate that Phase 2 mechanism atlas applies to real AI
workloads.

------------------------------------------------------------------------

# End-State of Phase 2

You will possess:

-   Structured understanding of Linux scheduler behaviour under GPU
    load.
-   VM subsystem interaction under CUDA allocation patterns.
-   Driver submission scaling characteristics.
-   Concurrency amplification effects.
-   Real workload validation.

This directly enables Phase 3 abstraction design.

