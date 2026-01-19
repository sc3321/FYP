# Interim Report Action Plan (Mon → Thu 13:00)

> Purpose: produce a DoC-aligned interim report with a clear Phase‑1 narrative and a practical Phase‑2/3 direction.

---

## Timeline

### Monday (today): lock structure + write the skeleton
- Create report skeleton with all headings (see below).
- Write **Introduction (1–2 pages)** in one sitting (draft quality is fine).
- Write Background **2.1 + 2.2**.
- Set up references in Vancouver format (Zotero or BibTeX) and cite as you write.

**End-of-day target:** 4–6 pages written + headings complete.

### Tuesday: Background bulk (most marks)
- Write Background **2.3–2.6**.
- Add citations while writing (avoid “cite later”).
- Write a short **Synthesis/Key takeaways** subsection that motivates the Phase‑1 axes.

**End-of-day target:** Background mostly complete (≈10–15 pages).

### Wednesday: finish remaining sections + polish
- Write **Project Plan (1–2 pages)**.
- Write **Evaluation Plan (1–2 pages)**.
- Optional: **ResNet validation** only if it runs cleanly in a short window (one model, one sweep). If not, describe it as planned work.

**End-of-day target:** full draft + references + figure placeholders.

### Thursday morning: tighten and submit
- One clarity pass (structure, flow, signposting).
- Check citations for all factual claims.
- Vancouver reference list formatting + final proofread.

---

## Report Structure (DoC-aligned)

### 1. Introduction (1–3 pages)
Suggested flow:
- Motivation: AI workloads, GPUs, and why OS interaction matters
- Project aim and approach: constraint discovery → mechanism isolation → abstraction prototype
- Interim progress summary (short)

### 2. Background (10–20 pages)
You can adjust headings later; this is a strong starting shape:

**2.1 AI workloads and datacenter reality**  
Training vs inference, throughput vs tail latency, multi-tenant pressures.

**2.2 GPU execution stack: what the OS sees**  
Framework → runtime → driver, kernels/streams/sync/memory residency.

**2.3 OS visibility boundary**  
Syscalls as interface events; why GPU work is opaque at the OS boundary.

**2.4 Prior systems for GPU scheduling/sharing (ML-focused)**  
A few high-quality exemplars (not a broad list).

**2.5 LithOS (core related work)**  
What it argues, what abstraction it proposes, where it fits vs this project.

**2.6 Synthesis: what’s missing**  
Key takeaways; motivate why Phase‑1 axes are the right starting point.

### 3. Project Plan (1–2 pages)
- Progress to date (bullet list)
- Phase 2 plan: mechanism-isolation experiments (interventions)
- Phase 3 plan: user-space abstraction prototype
- Fallbacks + extensions

### 4. Evaluation Plan (1–2 pages)
- Metrics: syscall categories + time; later add contention metrics (context switches, CPU time split, tail latency)
- Axes: **bytes, iterations, execution model**
- Success criteria: invariances/divergences; real workload validation (ResNet)

### 5. Ethical Issues (1–2 pages if necessary)
Likely short: no human subjects/personal data; brief note on datacenter energy/utilisation implications.

### 6. References (Vancouver)

---

## Where Phase‑1 Data Goes (lightweight, DoC-friendly)
- Include **1 small table/figure** (or a short paragraph) in **Project Plan** (“progress to date”) or **Evaluation Plan**.
- Keep it focused on **constraints**, not explanation:
  - kernel regime: OS-visible syscall structure largely invariant vs bytes/iters
  - execution axis: CPU vs CUDA shows different syscall footprint
  - implication: OS observes runtime lifecycle events, not GPU work granularity

---

## Optional ResNet (only if quick)
- One model, one sweep (batch size easiest).
- Goal: validate Phase‑1 constraints on a real workload.
- If it becomes a rabbit hole, skip and describe as planned validation.

