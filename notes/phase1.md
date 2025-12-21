# Phase 1 Timeline (Now → Jan 22)

**Phase 1 goal:** Build empirical grounding by observing how real GPU workloads (microbenchmarks and AI inference) interact with the Linux OS across NVIDIA and AMD GPUs. This phase is about *understanding reality*, not designing solutions yet.

---

## Week 1: Foundations & controlled microbenchmarks

### Focus
Establish a clean experimental harness and validate that you can reliably observe GPU–OS interactions.

### Tasks
- Finalize and clean up GPU program variants:
  - CPU-only baseline
  - GPU alloc/free only
  - GPU alloc + simple kernel
- Ensure versions exist for:
  - CUDA (NVIDIA)
  - HIP (AMD)
- Set up repeatable run scripts (fixed sizes, warmup, iterations).
- Run initial syscall profiling:
  - `strace -c`
  - Focused traces (`mmap`, `ioctl`, `futex`, `poll`).

### Deliverables
- Working repo with scripts and variants
- First syscall summary tables
- Short notes: *what surprised you, what didn’t*

### Skills developed
- GPU runtime basics
- Linux syscall tracing
- Experimental hygiene

---

## Week 2: Cross-vendor comparison (microbenchmarks)

### Focus
Identify which OS interactions are vendor-agnostic versus vendor-specific.

### Tasks
- Run the same microbenchmarks on:
  - NVIDIA lab machines
  - AMD GPU (cloud or borrowed)
- Normalize parameters (problem size, iteration count).
- Compare syscall distributions side-by-side.

### Deliverables
- NVIDIA vs AMD syscall tables
- Annotated list:
  - Common syscalls
  - Divergent behavior

### Skills developed
- Comparative systems analysis
- Hypothesis formation from data

---

## Week 3: Minimal Apple Metal exploration (bounded scope)

### Focus
Briefly contrast Linux GPU behavior with a tightly integrated platform.

### Tasks
- Run a minimal Metal compute workload locally.
- Capture one Metal System Trace.
- Record *qualitative* observations only.

### Deliverables
- 3–5 bullet-point observations
- One short subsection for interim report

### Skills developed
- Platform comparison literacy

---

## Week 4: Real AI workload on NVIDIA and AMD

### Focus
Move from synthetic benchmarks to datacenter-relevant workloads.

### Tasks
- Select one portable AI inference workload (e.g., ResNet-18 or small transformer).
- Run inference on:
  - CUDA (NVIDIA)
  - HIP/ROCm (AMD)
- Collect syscall profiles and basic latency metrics.
- Compare syscall patterns across vendors.

### Deliverables
- Syscall comparison tables for AI inference
- Notes on which OS interactions scale with workload complexity

### Skills developed
- ML workload profiling
- Understanding runtime pressure on the OS

---

## Weeks 5–6: Synthesis & interim report writing

### Focus
Turn experiments into a coherent narrative.

### Tasks
- Write ~2 hours/day on the interim report.
- While writing, run small follow-up experiments as needed.
- Synthesize:
  - What OS mechanisms appear essential?
  - What varies by vendor?
  - What seems accidental or historical?

### Deliverables
- Interim report submission
- Clear list of Phase 2 research questions

### Skills developed
- Technical writing
- Systems-level reasoning

