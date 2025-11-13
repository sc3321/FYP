# 🧭 FYP Phase 1 — Foundations and Bottleneck Discovery (Nov–Jan)

## 🎯 Objective

By **January (~8–10 weeks)**, develop:

- A deep, data-driven understanding of how AI workloads interact with OS abstractions.  
- A reproducible setup for measuring and profiling CPU/GPU performance.  
- A clear map of where the OS limits performance, latency, or predictability.  
- A specific, evidence-backed hypothesis — the **bottleneck** you’ll target with a new abstraction or kernel-level redesign.

---

## 🧩 Environment, Research Setup & Early Baselines (Weeks 1–2)

**Goals**
- Have a complete, working development environment.  
- Begin measuring and reading in parallel.

**Actions**
- ✅ Confirm VM or DoC remote environment works reliably.  
- ✅ Initialize repository structure:
  ```bash
  fyp-os-ai/
    notes/
    experiments/
    microbenches/
    kernel/
    scripts/
  ```
- Add a `scripts/capture_env.sh` that logs kernel version, CPU/GPU, drivers, and environment variables.  
- Start your **lab notebook** (`notes/lab-YYYY-MM-DD.md`) — one entry per session.  
- Begin **research reading loop** (2–3 papers/week):
  - OS abstractions for accelerators (GPUfs, Zorua, Aero).  
  - Scheduling and resource management in ML systems (Salus, Gandiva, Shenango, Caladan).  
  - For each: record *problem, hypothesis, evaluation method, takeaway*.  
- Begin **baseline microbenchmarks** (since you’re ready now):  
  - `syscall_cost.c`: measure getpid/read loop cost.  
  - `ctx_switch.c`: two threads ping-pong over a pipe.  
  - `page_faults.c`: 4K vs huge pages first-touch latency.  
  - GPU: measure pinned vs pageable H2D/D2H bandwidth, kernel launch overhead.  
- Record all data and plots in `experiments/baselines/`.  
- Start sketching your **“Contact Map”** — where your workload touches OS subsystems (syscalls, memory, scheduling, I/O).

---

## ⚙️ System Characterization and Profiling (Weeks 3–4)

**Goals**
- Understand OS behavior under real workloads, not just synthetic benchmarks.

**Actions**
- Build a simple **inference microservice** (Python or C++):
  - One small model (e.g., MLP or ResNet).
  - Local request serving + load generator.
- Profile using:
  ```bash
  strace -c python infer.py
  sudo perf stat -e cycles,instructions,context-switches,page-faults python infer.py
  sudo trace-cmd record -e sched_switch -e sys_enter -e sys_exit python infer.py
  ```
- GPU profiling with **Nsight Systems**:
  - Measure kernel launch intervals and H2D/D2H overlap.
- Record p50/p95/p99 latency, CPU utilization, syscalls/sec, page faults, and context switches.
- Save plots + notes in `experiments/inference-baseline/`.
- Write one clear summary:
  > “What parts of the OS are on the critical path for inference latency?”

---

## 🧠 Pattern Discovery and Hypothesis Formation (Weeks 5–6)

**Goals**
- Move from observation to insight — identify bottlenecks and mismatched abstractions.

**Actions**
- Review all results and ask:
  - Where does time accumulate inside the kernel?
  - What OS mechanism contributes most to unpredictability or tail latency?
- Translate findings into **candidate hypotheses**:

| Observation | Possible OS Limitation | Potential Redesign |
|--------------|-----------------------|--------------------|
| High syscall cost | Excessive user–kernel crossings | Batched async syscalls |
| GPU helper threads delayed | Scheduler unaware of GPU semantics | GPU-aware scheduling class |
| Page faults on tensor allocs | Dynamic VM model | Pinned / pre-faulted memory |
| Copy-heavy I/O path | Redundant buffering | Zero-copy or kernel-bypass path |

- For each, write a 1-page mini-proposal:
  - Symptom, suspected subsystem, intuition, and possible fix.  
- Narrow down to 2–3 realistic ideas for deeper exploration.

---

## 🔬 Early Prototyping & Validation (Weeks 7–8)

**Goals**
- Validate ideas quickly and safely in user space.  
- Measure whether they have measurable impact.

**Actions**
- Implement small prototypes:
  - `LD_PRELOAD` wrappers for `malloc`, `mmap`, or `mlock` (simulate pinned memory).  
  - Use **io_uring** to batch syscalls.  
  - Experiment with **SCHED_FIFO** or manual thread pinning for GPU feeder threads.
- Profile before/after with:
  ```bash
  perf stat -e cycles,context-switches,page-faults ./your_app
  sudo trace-cmd record -e sched_switch ./your_app
  ```
- Document improvements or neutral outcomes — both are valuable.  
- Commit reproducible scripts and summary tables under `experiments/prototypes/`.

**Deliverable by January**
- A reproducible measurement setup.  
- A detailed “OS contact map” for your AI workload.  
- Evidence of at least one concrete bottleneck.  
- A strong, data-driven hypothesis for your Phase-2 kernel work.

---

## 📚 Supporting Research Component

**Reading cadence**
- ~2 papers per week alternating between:
  - OS design & kernel abstraction (Exokernel, Barrelfish, Shenango, Caladan).  
  - AI system & GPU resource papers (Gandiva, Salus, E3, Zorua).  
- Capture for each: *problem → abstraction → method → results → lessons*.

**Goal**
By January:
- Understand how researchers justify new abstractions.
- Be able to describe your own OS bottleneck with research-grade reasoning.

---

## 🧭 Visual Flow Summary

```
Setup + Reading
   ↓
Microbenchmarks (syscalls, ctx switch, memory, GPU)
   ↓
Profile a small AI workload end-to-end
   ↓
Identify where OS abstractions limit performance
   ↓
Form hypotheses (what to redesign)
   ↓
Prototype & measure early fixes
   ↓
→ January Milestone: Select one OS bottleneck for kernel-level exploration
```

---

## ✅ Expected State by January

- Stable development + measurement environment.  
- Organized experimental data and notes.  
- First “Abstraction Contact Map” between OS and AI stack.  
- A single, well-motivated bottleneck ready for kernel or LSP-level redesign in Phase 2.
