# Phase 1: Detailed Plan (Next 3 Weeks)

**Project:** New OS Abstractions for AI Workloads in Datacenters
**Time window:** ~3 weeks (end of Dec → Jan 22)

---

## 0. Purpose of Phase 1 (anchor)

Phase 1 is a **constraint–discovery phase**. Its goal is *not* to design or implement a new OS abstraction, but to determine:

> **Which dimensions of AI-style GPU execution actually drive OS interaction, and which do not.**

The output of Phase 1 is a *small set of empirically grounded constraints* that any future OS/runtime abstraction must satisfy.

---

## 1. Core experimental axes (non‑negotiable)

Phase 1 reasoning is organised around three independent axes:

1. **Size axis (bytes)** – how much data is touched
2. **Repetition axis (iterations)** – how often work is issued
3. **Execution axis (execution model)** – *who* executes the work

   * CPU baseline (no GPU runtime)
   * GPU runtime bookkeeping only (alloc)
   * GPU runtime + kernel dispatch (kernels)

All Phase‑1 questions, analyses, and claims must map directly to one of these axes.

---

## 2. Workload consistency rules (important)

To make the execution axis meaningful:

* **Baseline and kernel workloads must be semantically comparable**

  * Same buffer size (`bytes`)
  * Same notion of repetition (`iters`)
  * Same unit of work (e.g., touch/increment array)

* **Alloc workload is intentionally different**, but:

  * Uses the same `bytes`
  * Uses `iters` as number of alloc/free (or alloc/copy) operations

This ensures differences reflect *execution model*, not workload semantics.

---

## 3. The three frozen Phase‑1 questions

These are **axis‑derived questions**, not data‑derived questions.

### Q1 – Size axis

> *Holding repetition and execution model constant, how do syscall categories change as buffer size increases?*

### Q2 – Repetition axis

> *Holding buffer size and execution model constant, which syscall categories scale with repetition and which remain constant?*

### Q3 – Execution axis

> *For fixed buffer size and repetition, which syscall categories appear or dominate only when GPU runtimes are involved compared to a CPU baseline?*

Once written, these questions are **frozen** for Phase 1.

---

## 4. How questions become claims (mechanical process)

For **each** frozen question:

1. Fix all *non‑relevant* axes
2. Collapse raw syscalls → syscall categories
3. Observe scaling behaviour (flat / linear / sublinear)
4. Rewrite the answer as a **constraint**:

> *OS interaction depends on A but is insensitive to B, ruling out C.*

Claims are not explanations; they are **eliminations of possibility**.

---

## 5. Week‑by‑week plan

### Week 1 – Question freezing & HIP‑CPU compression

**Goals:**

* Make the execution axis valid
* Freeze questions
* Produce first‑pass claims

**Tasks:**

* Implement real CPU baseline workload
* Ensure `bytes` and `iters` semantics are identical across variants
* Answer Q1–Q3 using HIP‑CPU data only
* Produce:

  * 1 table/plot per question
  * 1 paragraph answer per question
  * 1 draft claim per question

**Deliverables:**

* Frozen Q1–Q3
* 3 answer artifacts
* 3 draft constraints

---

### Week 2 – Real GPU & vendor validation

**Goals:**

* Test whether Phase‑1 constraints are structural or runtime‑specific

**Tasks:**

* Run same microbenchmarks on:

  * CUDA (NVIDIA)
  * HIP on real AMD GPU (if available)
* Re‑answer Q1–Q3 on the new stack
* Compare against HIP‑CPU results

**Deliverables:**

* Vendor comparison tables
* Claims annotated as:

  * Vendor‑agnostic
  * Vendor‑specific
  * Uncertain

---

### Week 3 – ResNet validation & Phase‑1 synthesis

**Goals:**

* Show relevance to real AI workloads
* Finalise Phase‑1 outputs

**Tasks:**

* Run ResNet inference under `strace -c`
* Apply same syscall categorisation
* Map ResNet behaviour to Phase‑1 claims
* Write Phase‑1 synthesis and Phase‑2 handoff

**Deliverables:**

* ResNet validation section
* Final set of 3–6 Phase‑1 constraints
* Clear statement of what Phase 2 must investigate

---

## 6. End‑of‑Phase‑1 success criteria

Phase 1 is complete when you can state, without CSVs:

> *Which axes of AI execution drive OS interaction, which do not, and which behaviours are structural across vendors and workloads.*

