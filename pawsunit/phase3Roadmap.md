# Phase 3 Final Roadmap (43 Days)

## Objective

Deliver a **minimal, well-evaluated OS-adjacent abstraction**:

> Phase-scoped GPU work-unit abstraction enabling LC/BE-aware policy

Core deliverables:
- Userspace C/C++ library (phase abstraction)
- Semantic event log
- Multiprocess policy layer (BE pacing)
- Microbenchmark evaluation (primary)
- vLLM validation (secondary)
- High-quality report

---

# Guiding Principles

- Narrow scope, deep execution
- One abstraction, one policy, one evaluation axis
- Evidence > complexity
- Clean argument > feature count

---

# Week 1 (Days 1–7): Spec + Foundation

## Goals

- Lock abstraction semantics
- Build minimal usable library
- Validate phase ↔ OS signal alignment

## Tasks

### 1. Write abstraction spec (2–3 pages)
Define:
- What is a phase?
- Phase lifecycle (begin/end)
- Workload classes (LC / BE)
- Metadata schema
- Non-goals

### 2. Implement core library (libphase)

API:
```c
gpu_phase_begin(class, phase_type);
gpu_phase_end();
```

Internals:
- timestamp (monotonic)
- pid/tid
- phase_id
- write structured log (JSONL or binary)

### 3. Build single-process microbenchmark

- Simple CUDA workload
- Wrap with phase markers

### 4. Validation

- Run with strace
- Confirm:
  - phase boundaries align with syscall bursts
  - sync phases align with futex/poll windows

---

# Week 2 (Days 8–14): Multiprocess Baseline

## Goals

- Establish LC vs BE interaction baseline
- No policy yet

## Tasks

- Two processes (LC + BE)
- Event logging
- Measure LC latency + BE throughput
- Visualise overlap

---

# Week 3 (Days 15–21): Policy Layer

## Goals

- Implement LC-aware BE pacing
- Demonstrate controllable tradeoff

## Tasks

- Policy process (socket or shared file)
- BE delay sweep (0us → 5ms)
- Measure latency vs throughput

---

# Week 4 (Days 22–28): Evaluation Depth

## Goals

- Strengthen evidence

## Tasks

- Kernel size variation
- Concurrency variation
- Overhead measurement
- Failure case analysis

---

# Week 5 (Days 29–35): vLLM Validation

## Goals

- Show abstraction relevance

## Tasks

- LC vs BE clients
- Request-level phase annotation
- Compare patterns with microbenchmarks

---

# Week 6 (Days 36–43): Report + Finalisation

## Tasks

- Write report
- Clean plots
- Reproducibility
- Final polish

---

# Engineering Growth Areas

## High-value skills

- Abstraction design
- Systems boundary reasoning
- Multiprocess coordination
- Performance evaluation
- C/C++ systems programming

## Use help for

- Boilerplate code
- Plotting
- Report structure

## Do yourself

- Abstraction thinking
- Experiment design
- Result interpretation

---

# Success Criteria

- Clear missing semantic identified
- Minimal abstraction defined
- Measurable LC/BE improvement shown
- Clean evaluation
- Honest limitations

