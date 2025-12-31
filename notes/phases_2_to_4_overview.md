# Phases 2–4: Purpose, Continuity, and the Role of Implementation

This document describes **Phases 2–4** of the project in a way that is consistent, unambiguous, and aligned with the project title:

> **New OS abstractions for AI workloads in datacenters**

The emphasis is on *intent* rather than tooling, and on *why* each phase exists.

---

## Phase 1 → Phase 2: The handoff (recap)

**Phase 1 output:**

* A small set of empirically grounded **constraints** describing how OS interaction behaves with respect to:

  * workload size (bytes)
  * repetition / submission frequency (iters)
  * execution model (CPU baseline vs GPU runtime vs GPU execution)
* These constraints explicitly rule out classes of explanations and optimisations.

**Key handoff principle:**

> Phase 1 tells us **where** OS abstractions are stressed. Phase 2 exists to understand **why** those stresses arise.

---

## Phase 2: Why-analysis and abstraction pressure

### Core purpose

Phase 2 answers the question:

> **Why do the Phase-1 constraints arise, and what *kind* of abstraction is missing because of them?**

This phase is about **mechanism isolation**, not about proposing solutions yet.

---

### How Phase 2 differs from Phase 1

Although Phase 2 may reuse similar tooling (strace, microbenchmarks), the *intent* is fundamentally different.

| Phase 1            | Phase 2               |
| ------------------ | --------------------- |
| Observational      | Interventional        |
| Vary inputs        | Restructure execution |
| Discover scaling   | Test causality        |
| Syscall categories | Syscall *roles*       |

Phase 2 experiments are explicitly designed *because of* Phase-1 constraints.

---

### What the "why" means in practice

Given a Phase-1 constraint such as:

* *OS interaction scales with repetition rather than data size*

Phase 2 asks:

* Is this due to kernel launch granularity?
* Synchronisation frequency?
* Runtime bookkeeping structure?

To answer these, Phase 2 uses **targeted microbenchmarks** that:

* batch or fuse work submissions
* move synchronisation points
* alter allocation lifetimes

If OS interaction changes in the predicted way, the underlying mechanism is confirmed.

---

### Phase-2 outputs

* Syscalls grouped by **role** (e.g., dispatch, synchronisation, bookkeeping)
* Identification of **abstraction pressure**, such as:

  * OS sees too many fine-grained GPU events
  * OS cannot reason about GPU work as a unit
* Clear articulation of *what existing OS/runtime abstractions fail to express*

These outputs define the **design space** for Phase 3.

---

## Phase 3: Abstraction formulation and prototyping (clarified)

### Core purpose

Phase 3 answers:

> **What would a cleaner OS-facing abstraction look like, given the pressures identified in Phase 2?**

This is where *implementation* appears — but only as a means of forcing semantic precision.

---

### What Phase 3 implementation is (and is not)

**Phase 3 is not:**

* rewriting the OS kernel
* reimplementing CUDA or HIP
* building a production runtime

**Phase 3 is:**

* making the abstraction *concrete enough to test*

---

### Required Phase-3 work

1. **Abstraction definition**

   * Define what the abstraction represents (e.g., a GPU job, execution phase, or batch)
   * Specify its lifecycle and semantics
   * Explain what existing abstractions fail to capture

2. **User-space prototype (C/C++)**

   * Implement a *minimal* C/C++ library that embodies the abstraction
   * Integrate it *above* CUDA/HIP (existing calls flow through it)
   * Use it to restructure execution according to the abstraction

3. **Empirical demonstration**

   * Run microbenchmarks and AI workloads through the prototype
   * Show that Phase-1 constraints are mitigated, reshaped, or better explained

The prototype is a **model of the abstraction**, not its final system placement.

---

### Optional extension (stretch goal only)

If time and clarity permit:

* Implement a *minimal* kernel-facing mock or module
* Use it only to demonstrate that the abstraction could live at the OS boundary

This is **not required** and should not be attempted unless the abstraction is already well-defined.

---

### Phase-3 outputs

* Precise abstraction specification (API + semantics)
* Working user-space prototype
* Evidence that the abstraction addresses Phase-1 constraints

---

## Phase 4: Evaluation

### Core purpose

Phase 4 evaluates:

> **Does the proposed abstraction meaningfully improve OS interaction for AI workloads in datacenters?**

---

### Evaluation dimensions

* **Effectiveness:** reduces syscall overhead or undesirable scaling
* **Generality:** holds across vendors and workloads
* **Robustness:** behaves predictably under load and variation

---

### Workloads

* Controlled microbenchmarks
* Real AI workloads (e.g., ResNet inference)

---

### Phase-4 outputs

* Quantitative comparison:

  * baseline runtime
  * restructured / abstracted execution
* Discussion of trade-offs and limitations
* Implications for OS design in datacenter AI systems

---

## Project arc (one paragraph)

Phase 1 identifies *where* OS abstractions are stressed by AI workloads. Phase 2 explains *why* those stresses arise by isolating mechanisms. Phase 3 formulates and prototypes a cleaner abstraction that captures the missing semantics. Phase 4 evaluates whether this abstraction is effective, general, and relevant for datacenter-scale AI workloads.

This progression directly supports the project goal of proposing **new OS abstractions for AI workloads in datacenters**.

