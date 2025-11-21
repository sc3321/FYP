# Phase 1 Action Plan (Now → Mid-January)

---

# 🎯 Phase 1 Goals

By mid‑January :

* A solid mental model of **GPU programming** (HIP/CUDA fundamentals).
* An understanding of how **GPU runtimes** are structured and implemented.
* A clear map of the **runtime → driver → OS → hardware** boundary.
* Early signs of **OS abstraction mismatches** worth investigating.
* The foundation required for real hardware + systems experiments in Phase 2.

GPU programming fundamentals + runtime comprehension is the priority.

---

# 📅 Week-by-Week Plan (Concise + Detailed)

Parallel tracks:

* **Track A:** GPU programming concepts (HIP/CUDA)
* **Track B:** GPU runtime architecture (LithOS + ROCm)
  Both build toward identifying OS interaction points and future bottlenecks.

---

## **Week 1 — GPU Programming Fundamentals + Runtime Shape (Light)**

### Tasks

**Track A:**

* Learn the core GPU model: kernels, blocks/grids, threads.
* Write a minimal HIP program and run with HIP CPU backend.
* Run a simple CUDA kernel on a lab NVIDIA GPU.

**Track B:**

* Read high-level documentation for LithOS or similar small GPU runtime.
* Understand conceptually: queues, memory, dispatch.

### Why This Matters

Start forming the **mental shape** of both the *programming model* and the *runtime model*.
These two worlds must be understood together to reason about OS abstractions later.

---

## **Week 2 — OS Interactions + First Runtime Tracing (Light)**

### Tasks

**Track A:**

* Use `strace -c` on a simple CUDA program.
* Observe key syscalls: `mmap`, `mlock`, `futex`, `clone`, `ioctl`.

**Track B:**

* Produce a rough “stack diagram”: App → Runtime → Driver → OS → Hardware.
* Add notes on LithOS’s OS interactions (even superficially).

### Why This Matters

Begin seeing that GPU execution is **not magic** — it drives real OS mechanisms.
This becomes the foundation of understanding where OS abstractions might fail.

---

## **Week 3 — Pre‑Christmas Supervisor Collaboration + LithOS Understanding**

### Tasks

**Track A:**

* Expand HIP/CUDA understanding: host/device memory, transfers, pinned memory concepts.

**Track B:**

* Skim LithOS code focusing on:

  * memory management
  * command queues
  * OS-facing calls
* Produce a one-page summary of LithOS architecture.

**Supervisor Check‑In:**

* Present:

  * GPU model basics
  * High-level runtime understanding
  * Your initial contact map
* Get direction for Christmas work.

### Why This Matters

This meeting is **crucial** for alignment.\

---

## **Week 4 — HIP Runtime & Programmer View (Start of Break)**

### Tasks

**Track A:**

* Explore HIP API surface: memory ops, streams, events.
* Write small HIP CPU-mode tests (hipMalloc, hipMemcpy, simple kernels).

**Track B:**

* Inspect how the HIP runtime structures its host-side logic.
* Understand HIP’s fallback mechanisms without hardware.

### Why This Matters

HIP is the **front door** to ROCm.
Understanding it prepares me to read ROCm internals and interpret its behaviour.

---

## **Week 5 — ROCm Runtime Internals (Mid Break)**

### Tasks

**Track A:**

* Compare HIP CPU-mode behaviour with CUDA behaviour.

**Track B:**

* Read ROCm HIP runtime source (host side): memory, streams, launch paths.
* Note every syscall or OS interaction path in code.

### Why This Matters

This is first exposure to industrial-grade GPU runtime architecture.
You see how real runtimes depend on Linux: memory, threads, ioctls.

---

## **Week 6 — OS Boundaries & Early Hypotheses (Late Break)**

### Tasks

**Track A:**

* Write microbenchmarks in CPU-only HIP:

  * hipMalloc timing
  * hipMemcpy timing
  * kernel launch overhead (CPU-mode)

**Track B:**

* Expand OS ↔ GPU contact map:

  * where runtimes rely on Linux VM
  * where threads and scheduling come in
  * how command queues interact with kernel drivers
* Draft 1–2 early hypothesis areas.

### Why This Matters

Begin connecting all knowledge into clear **problem areas** — the seeds of Phase 2.

---

## **Week 7 — Consolidation & Hardware Planning (Return to Uni)**

### Tasks

**Track A:**

* Consolidate GPU programming understanding.

**Track B:**

* Write a concise Phase-1 summary:

  * programming model
  * LithOS + ROCm architecture
  * OS boundary map
  * mismatch hypotheses
* Investigate AMD GPU access:

  * department resources
  * cloud MI210/MI250 instances
  * AMD academic programs

### Why This Matters

Coherent technical story.
**Hardware environment** for Phase 2.

---

## **Week 8 — Phase 2 Direction Setting (Mid-January)**

### Tasks

* Finalise hypothesis list.
* Select the most promising OS–GPU abstraction direction.
* Plan first hardware-based experiments.
* Begin preparing small runtime/OS instrumentation code.

### Why This Matters

Enter Phase 2 with:

* clarity
* technical depth
* architectural understanding
* a viable experimental plan

This is the transition from learning → innovation.

---

# 🎓 End-of-Phase 1 Outcomes

* Understand GPU programming and runtimes in parallel.
* Have a detailed OS ↔ GPU boundary map.
* Understand LithOS (conceptual) and ROCm HIP runtime (industrial).
* Have early mismatch hypotheses for OS abstractions.
* Be ready for real GPU hardware experiments in Phase 2.

