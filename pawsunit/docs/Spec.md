# Phase 3 Abstraction Specification

## Phase-Scoped GPU Work-Unit Abstraction

---

## 1. Motivation

Phase 2 demonstrated that GPU execution appears at the operating system boundary only through coarse-grained signals such as:

* driver interactions (e.g., `ioctl`)
* host blocking and wake events (e.g., `futex`, `poll`)
* memory management activity (e.g., `mmap`)

While these signals expose **that** interaction occurs, they do not expose **what that interaction represents** in terms of program execution.

In particular, the OS cannot infer:

* which GPU activity corresponds to a single logical unit of work
* whether activity is latency-critical or best-effort
* when a meaningful execution boundary begins or ends

This lack of semantic structure prevents the OS (or OS-adjacent policy) from reasoning about interference, prioritisation, or scheduling decisions.

---

## 2. Objective

Define a **minimal abstraction** that introduces semantic structure at the host–device boundary, without modifying the GPU runtime or operating system.

The abstraction must:

* group GPU-related activity into **explicit execution phases**
* associate phases with a **workload class** (e.g., LC vs BE)
* expose this information to an adjacent **policy layer**

---

## 3. Core Abstraction

### 3.1 Definition

A **phase-scoped GPU work unit (phase)** is defined as:

> A contiguous region of host-side execution corresponding to a logically meaningful unit of GPU-related work, explicitly delimited by the program.

Each phase has:

* a **start boundary**
* an **end boundary**
* an associated **workload class**
* associated **metadata**

---

### 3.2 Lifecycle

A phase follows a strict lifecycle:

```text
PHASE_BEGIN → (GPU submission / execution / synchronisation) → PHASE_END
```

Semantics:

* `PHASE_BEGIN` marks the start of a logical unit of work
* `PHASE_END` marks its completion from the host perspective
* All host–device interactions within this region are attributed to the phase

---

## 4. Workload Classification

Each phase is assigned a **workload class**:

```text
LC (Latency-Critical)
BE (Best-Effort)
UNKNOWN (optional fallback)
```

### 4.1 Semantics

* **LC phases** represent latency-sensitive work (e.g., inference requests)
* **BE phases** represent throughput-oriented work (e.g., background computation)
* Classification is **provided by the application**, not inferred

---

## 5. Interface

### 5.1 API

The abstraction is exposed via a minimal userspace API:

```c
void gpu_phase_begin(int class, int phase_type);
void gpu_phase_end(void);
```

### 5.2 Requirements

* Calls must be **paired** (no nesting in initial design)
* Calls must be **low overhead**
* Calls must be safe in multi-threaded contexts

---

## 6. Event Model

Each phase emits structured events:

### 6.1 Event Types

```text
PHASE_BEGIN
PHASE_END
```

### 6.2 Event Fields

Each event contains:

```text
timestamp      (monotonic clock)
pid            (process ID)
tid            (thread ID)
phase_id       (unique identifier)
class          (LC / BE / UNKNOWN)
phase_type     (optional, e.g., request, batch)
```

### 6.3 Semantics

* Events form a **partial ordering** across processes
* Events allow reconstruction of:

  * phase durations
  * phase overlap
  * LC vs BE concurrency

---

## 7. Policy Layer Interface

The abstraction is designed to enable an external **policy layer**.

### 7.1 Inputs

* stream of phase events across processes

### 7.2 Derived State

The policy layer can compute:

```text
LC_active = true if any LC phase is active
BE_active = true if any BE phase is active
overlap   = LC ∧ BE
```

### 7.3 Example Policy

A minimal policy enabled by this abstraction:

```text
If LC_active:
    delay or throttle BE work
Else:
    allow BE to run normally
```

---

## 8. Design Constraints

The abstraction is intentionally constrained:

### 8.1 Non-goals

This abstraction does **not**:

* schedule GPU kernels
* modify CUDA or driver behaviour
* provide fine-grained kernel visibility
* guarantee performance improvements in all cases

### 8.2 Assumptions

* Applications can identify meaningful phase boundaries
* Phase classification (LC/BE) is available at the application level
* Host-side boundaries approximate device-side behaviour sufficiently for policy use

---

## 9. Limitations

* Phase boundaries are **coarse-grained** relative to GPU execution
* No visibility into:

  * individual kernel execution
  * device scheduling decisions
* Effectiveness depends on correctness of phase annotation
* No direct control over GPU execution (only indirect via pacing)

---

## 10. Positioning

This abstraction sits:

```text
Application
    ↓ (phase annotation)
Userspace abstraction (this work)
    ↓ (event stream)
Policy layer
    ↓
Unmodified CUDA runtime / driver / OS
```

It provides a **semantic bridge**, not a replacement for runtime or OS scheduling.

---

## 11. Summary

This work introduces a minimal abstraction that:

* exposes **execution boundaries**
* attaches **workload intent**
* enables **policy reasoning across processes**

without requiring changes to:

* the operating system
* the GPU driver
* the ML framework

It is designed to test whether **modest semantic exposure** can improve host-level reasoning about GPU workloads.

