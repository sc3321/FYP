# Phases 2–4 Overview: Direction Beyond the Interim Report

This document gives a *high-level* view of where the project is heading after Phase 1. It is intentionally not over-specified; the exact direction should be guided by Phase 1 results.

---

## Phase 2: Abstraction discovery and stress testing

### Core goal
Distill Phase 1 observations into a small set of *essential OS abstractions* required by GPU workloads.

### What you will do
- Group syscalls by *role* rather than name:
  - Memory management
  - Submission/control
  - Synchronization
  - Scheduling
- Identify which patterns:
  - Appear across vendors
  - Scale with workload complexity
- Design targeted microbenchmarks to stress specific pressures (e.g., many small kernel launches, heavy synchronization).

### Key outputs
- A runtime–OS interaction map
- A short list of candidate abstractions the OS implicitly provides (or lacks)

### Skills developed
- Systems abstraction extraction
- Experimental refinement

---

## Phase 3: Abstraction design and prototyping

### Core goal
Propose and prototype a cleaner OS-facing abstraction for GPU execution.

### What you will do
- Design a small, explicit API that captures the essential semantics uncovered in Phase 2.
- Implement the abstraction primarily in user space as a C/C++ library.
- Integrate the library above CUDA/HIP to run real workloads through it.

### Optional extension (only if useful)
- Implement a minimal kernel module ("hidden gem") to demonstrate that the abstraction could live at the OS boundary.
- Use it only for toy experiments and conceptual validation.

### Key outputs
- Abstraction specification (API + semantics)
- Working prototype

### Skills developed
- API and systems design
- Low-level C/C++ programming

---

## Phase 4: Evaluation and reflection

### Core goal
Evaluate the proposed abstraction and reflect on its implications.

### What you will do
- Compare native execution vs execution through your abstraction:
  - Syscall counts
  - Context switches
  - CPU overhead
  - Qualitative complexity
- Discuss limitations and trade-offs.
- Reflect on generalization to other accelerators.

### Key outputs
- Evaluation chapter
- Clear conclusions tied to datacenter relevance

### Skills developed
- Systems evaluation
- Research synthesis and reflection

---

## Final note
You do not need to decide now how far into kernel space you will go. Phase 1 grounds the project in reality; Phase 2 tells you *what matters*; Phase 3 chooses the right level of implementation; Phase 4 judges it honestly.

