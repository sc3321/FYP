# Phase 3 Narrowing Logbook

## 0. Project state
- Date: 24/04.
- Current goal: Narrow down Phase 3 abstraction.
- Next deadline: 01/05 Supervisor meeting.
- Current main uncertainties:

---

## 1. Phase 2 pressure shortlist

### Pressure family 1
- **Branch/source:** Sync topology variation
- **What was observed:**  
  Under the slow case (sync at end of workload), there is helper-thread kernel-level visibility into waiting. There is repeated `poll()` activity on helper threads, including activity around eventfd and `/dev/nvidia0`, with occasional `futex` wakes. Under the per-iteration topology, there are consistent latency waits, but with minimal or no clear syscall-visible structure in the immediate sync windows.

- **Why it matters:**  
  Even when GPU completion becomes scheduler-visible, it is exposed only through generic waiting mechanisms such as `poll()` and `futex`, often distributed across helper threads. There is no explicit indication of *what* is being waited on or whether these waits correspond to meaningful logical work boundaries.

- **Why it felt strong:**  
  The contrast between slow-path visibility and fast-path opacity shows that GPU completion can be observable at the OS boundary without being semantically interpretable.

- **What host-visible mechanism seems involved:**  
  `poll()`, `futex`, helper-thread activity, eventfd, driver wakeups.

---

### Pressure family 2
- **Branch/source:** Submission topology variation
- **What was observed:**  
  Steady-state launches in the fragmented submission cases are individually cheap but collectively expensive. Each launch becomes smaller in kernel time and launch time, but the total accumulated launch cost increases significantly. Device execution is also fragmented into many smaller kernels, increasing total GPU kernel execution time. Final synchronization cost also increases with fragmentation, but it is not the dominant factor.

- **Why it matters:**  
  Fragmentation produces a dense stream of host-side launch activity, but the OS cannot determine whether these events correspond to one logical job, multiple independent tasks, or a single phase of a larger workload.

- **Why it felt strong:**  
  The workload/runtime clearly knows that these launches belong to a structured computation, but the OS only observes an unstructured stream of low-level activity.

- **What host-visible mechanism seems involved:**  
  CUDA launch path, host-side dispatch overhead, driver submission activity, `ioctl`.

---

### Pressure family 3
- **Branch/source:** Allocation lifetime variation
- **What was observed:**  
  When allocation lifetime changes from run-scoped to iteration-scoped, this becomes visible at the OS boundary as repeated allocation/deallocation activity, including bursts of driver interaction and memory-management activity.

- **Why it matters:**  
  The OS cannot distinguish between allocations intended to persist and be reused, and allocations intended to be short-lived or ephemeral.

- **Why it felt strong:**  
  The same low-level memory operations appear similar at the OS boundary despite reflecting different workload intent.

- **What host-visible mechanism seems involved:**  
  `mmap`, `munmap`, `mprotect`, driver `ioctl`, VM activity.

---

### Pressure family 4
- **Branch/source:** Scheduler pressure / GEMM fragmentation / CPU contention
- **What was observed:**  
  For compute-heavy GEMM kernels, increasing launch fragmentation from one large launch to many smaller launches significantly amplified the host-visible completion path. `futex` and `poll` activity increased substantially. Same-core CPU contention amplified this further, producing very large increases in wait time and wake activity. Sync placement had limited effect in the fragmented small-kernel case. Under extreme oversubscription, the system appeared to enter a regime dominated by synchronization churn and wake/sleep storms.

- **Why it matters:**  
  GPU execution progress depends partly on the timely scheduling of host-side runtime and completion-management threads. Fine-grained GPU work increases the number of completion events, which amplifies scheduler interaction. This can directly affect end-to-end completion time.

- **Why it felt strong:**  
  This branch revealed a structural contrast with the submission topology branch: lightweight fragmented kernels primarily expose dispatch accumulation, while heavier compute kernels expose completion and scheduler amplification. It also showed that host-side scheduling can become the primary bottleneck under contention.

- **What host-visible mechanism seems involved:**  
  `futex`, `poll`, scheduler wake/sleep behaviour, runqueue contention, helper-thread scheduling.

---

## 2. Candidate missing semantics

### Candidate A: Sync / completion semantics

- **Motivating pressure family:**  
  Sync topology variation.

- **What ambiguity exists at the host boundary:**  
  The OS observes generic waiting through mechanisms such as `poll()` and `futex`, but cannot determine what logical unit of GPU work is being waited on. It cannot tell whether a wait corresponds to completion of a meaningful phase, an iteration boundary, a request boundary, or incidental runtime/helper-thread behaviour.

- **What could the workload/runtime know that the OS does not:**  
  The workload or runtime may know whether synchronization is occurring per iteration, at the end of a full phase, or at the completion of a logical unit of work. It may also know whether the completion point is latency-critical or merely part of background progress.

- **One extra truthful bit I might want to export:**  
  A phase-level completion boundary: “this wait corresponds to completion of logical phase X.”

- **Rough semantic category:**  
  Completion / phase.

- **Why this might matter:**  
  It could allow an OS-adjacent policy layer to distinguish meaningful GPU completion from incidental runtime waiting. This would improve attribution and make completion pressure easier to reason about.

- **Why this might be wrong:**  
  This risks becoming mostly observational. If the exported completion information is not consumed by a policy, then it may amount only to better logging rather than a meaningful abstraction.

---

### Candidate B: Submission grouping / phase-scoped GPU work unit

- **Motivating pressure family:**  
  Submission topology variation and scheduler pressure.

- **What ambiguity exists at the host boundary:**  
  The OS observes a stream of launches, waits, helper-thread activity, and driver interactions, but cannot determine whether these events belong to one logical phase, multiple independent tasks, a latency-critical request, or best-effort background work.

- **What could the workload/runtime know that the OS does not:**  
  The workload or runtime may know which launches belong together, where a logical phase begins and ends, whether the phase is latency-critical or best-effort, and whether the work is part of a larger request, batch, or execution episode.

- **One extra truthful bit I might want to export:**  
  Logical phase boundaries and workload class, for example:
  - phase begin / phase end
  - latency-critical vs best-effort
  - request or phase identifier
  - possibly dispatch-heavy vs completion-heavy classification

- **Rough semantic category:**  
  Grouping / phase / criticality.

- **Why this might matter:**  
  This candidate directly addresses the strongest Phase 2 pressure. It can explain both dispatch-side fragmentation in the SAXPY submission experiments and completion-side amplification in the GEMM scheduler experiments. It gives an OS-adjacent policy layer the missing structure needed to attribute pressure and potentially treat latency-critical and best-effort GPU phases differently.

- **Why this might be wrong:**  
  If the prototype only records phase boundaries without changing any policy or behaviour, it may look like instrumentation rather than an abstraction. To be convincing, the phase metadata needs to be consumed by a simple policy, such as launch pacing, chunking, stream priority selection, CPU affinity control, or interference diagnosis.

---

### Candidate C: Allocation lifetime semantics

- **Motivating pressure family:**  
  Allocation lifetime variation.

- **What ambiguity exists at the host boundary:**  
  The OS cannot distinguish between ephemeral GPU-related allocations and persistent allocations intended for reuse. Repeated allocation and deallocation activity appears as low-level VM and driver interaction, without exposing the intended lifetime of the memory.

- **What could the workload/runtime know that the OS does not:**  
  The workload or runtime may know whether a buffer will be reused across iterations, whether it is temporary, or whether it belongs to a long-lived model/request state.

- **One extra truthful bit I might want to export:**  
  Allocation lifetime intent, such as persistent, phase-scoped, or ephemeral.

- **Rough semantic category:**  
  Memory / lifetime.

- **Why this might matter:**  
  If lifetime intent were visible, an OS-adjacent or runtime policy could reduce repeated allocation churn, improve reuse, or reduce VM pressure.

- **Why this might be wrong:**  
  This direction may turn into a CUDA memory allocator or memory-pooling optimisation rather than an OS-facing abstraction. It is useful as a secondary or future direction, but less directly connected to scheduler pressure and Phase 3 policy.

---

### Candidate D: Scheduler-aware workload classification

- **Motivating pressure family:**  
  Scheduler pressure / GEMM fragmentation / CPU contention.

- **What ambiguity exists at the host boundary:**  
  The OS cannot distinguish between CPU-paced GPU work, where dispatch accumulation dominates, and GPU-paced work, where completion and scheduler pressure dominate. It also cannot tell which host-side runtime/helper threads are on the critical path for GPU progress.

- **What could the workload/runtime know that the OS does not:**  
  The workload or runtime may know the rough structure of the GPU phase, the expected granularity of launches, whether the work is latency-critical or best-effort, and whether completion latency matters for end-to-end progress.

- **One extra truthful bit I might want to export:**  
  A workload or phase classification, such as:
  - latency-critical
  - best-effort
  - dispatch-heavy
  - completion-heavy
  - phase-scoped GPU work unit

- **Rough semantic category:**  
  Criticality / phase / scheduler relevance.

- **Why this might matter:**  
  This candidate gives a policy layer a reason to treat different GPU phases differently. For example, latency-critical phases could be protected from same-core contention, while best-effort fragmented phases could be chunked, paced, or routed differently.

- **Why this might be wrong:**  
  It may be difficult to define dispatch-heavy or completion-heavy robustly without profiling. This candidate is probably best merged into Candidate B rather than treated as a separate abstraction.

---

## Current narrowing decision

The strongest direction is **Candidate B: submission grouping / phase-scoped GPU work unit**, with Candidate D merged into it as the scheduler-policy dimension.

The emerging abstraction is:

> A phase-scoped GPU work-unit abstraction that exposes logical GPU execution boundaries and workload class to a host-side policy layer.

This abstraction is motivated by the fact that Phase 2 revealed two different fragmentation regimes:

- lightweight fragmented kernels mainly expose host dispatch accumulation
- heavier fragmented kernels mainly expose completion and scheduler amplification

In both cases, the OS sees low-level events without knowing which logical unit of work they belong to or how important they are. A phase-scoped abstraction could provide this missing structure.

A minimal prototype could therefore expose:

- `gpu_phase_begin(name, class)`
- `gpu_phase_end()`
- wrapped launch and synchronization calls
- optional policy behaviour based on phase class

The policy should not merely log the metadata. It should consume it in a simple way, for example:

- latency-critical phase: submit immediately, avoid throttling
- best-effort phase: chunk launches, yield between chunks, or reduce interference with latency-critical work

This makes the Phase 3 prototype more than instrumentation: it becomes a small model of how an OS-facing abstraction could support better attribution and scheduling policy.
---

## 3. vLLM validation plan

### Mechanism family 1: Submission / dispatch fragmentation

- **Microbenchmark source:** Submission topology variation.
- **Actual mechanism isolated:** Increasing the number of small GPU submissions amplifies host-side launch/driver activity, even when each individual launch is cheap.
- **Plausible vLLM analogue:** Many short requests with short prompts and short outputs, especially decode-heavy serving where work is repeatedly advanced in small steps.
- **What would count as supportive evidence:** Dense, repeated `ioctl`-dominated bursts or many small syscall clusters during request processing, with activity increasing under many short requests compared with fewer longer requests.
- **What would count as ambiguous evidence:** Increased syscall activity exists, but is dominated by networking, Python/event-loop behaviour, or unrelated `epoll_wait` rather than GPU-driver interaction.
- **What would count as weakening evidence:** Short-request workloads do not produce denser submission/driver activity than the long-request or baseline case; activity appears flat, sparse, or unrelated to request structure.
- **Scenarios to try:** Many sequential short-prompt, short-output requests; optionally repeat with concurrent short requests if sequential runs are too sparse.

### Mechanism family 2: Completion / scheduler pressure

- **Microbenchmark source:** Scheduler pressure / GEMM fragmentation / CPU contention.
- **Actual mechanism isolated:** Compute-heavier fragmented GPU work increases host-visible completion management, including `poll`, `futex`, helper-thread activity, and scheduler sensitivity under CPU contention.
- **Plausible vLLM analogue:** Long-prompt prefill and/or longer generation requests, especially under concurrent load or same-core CPU contention.
- **What would count as supportive evidence:** Longer or more frequent `poll`/`futex` intervals, increased helper-thread wait/wake activity, or stretched completion-related clusters compared with the uncontended baseline.
- **What would count as ambiguous evidence:** Waiting increases, but only in server/network/event-loop threads and cannot be plausibly connected to GPU completion or driver/helper-thread behaviour.
- **What would count as weakening evidence:** Long prompts, concurrency, or CPU contention do not change completion/waiting behaviour relative to short prompts or uncontended runs.
- **Scenarios to try:** Sequential long-prompt requests as a baseline, then concurrent long requests or mixed long+short requests, then repeat one case under CPU-core contention.

---

## 4. vLLM validation notes

### Run
- Date:
- Scenario:
- What pressure family I was probing:
- Tools used:
- What I observed:
- Does this look equivalent / weaker / stronger / different:
- Notes:

---

## 5. Implementation surface sketches

### Candidate A

#### Userspace wrapper version
- What object/API exists:
- What state is tracked:
- What semantics are exported:
- What changes structurally:

#### Interposition version
- What gets intercepted:
- What state must persist:
- What would be hard:

#### Evaluation visibility
- What eBPF / tracing would observe:
- What metrics would show change:

### Candidate B
(same structure)

---

## 6. eBPF notes

### What eBPF is
- My explanation in plain words:

### What it is good for in this project
- ...

### What it is not good for in this project
- ...

### Relevant events / attach ideas
- ...

### Questions I still have
- ...

---

## 7. Candidate comparison table

| Candidate | Motivating pressure | vLLM support | Missing semantic | Prototype shape | Evaluation path | Biggest risk |
|-----------|---------------------|--------------|------------------|-----------------|-----------------|--------------|
| A         |                     |              |                  |                 |                 |              |
| B         |                     |              |                  |                 |                 |              |
| C         |                     |              |                  |                 |                 |              |

---

## 8. PowerPoint storyboard

### Slide 1
- Purpose:
- Figure/table:
- Point to make:

### Slide 2
- Purpose:
- Figure/table:
- Point to make:

### Slide 3
- Purpose:
- Figure/table:
- Point to make:

...

---

## 9. End-of-day reflection
- What became clearer today:
- What got weaker today:
- What now seems most plausible:
- What still feels confusing:
- The single most important next step tomorrow:
