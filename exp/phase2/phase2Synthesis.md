# Phase 2 Synthesis Working Document

## Purpose

This document is the master synthesis note for the final closeout of Phase 2. Its role is to:

* gather the conclusions of each experimental branch in one place;
* record evidence, caveats, and open questions;
* extract defensible mechanism-level claims;
* prepare material for the tutor meeting;
* define the exact bridge into Phase 3.

---

## 1. Phase 2 status snapshot

### 1.1 Overall objective

Phase 2 exists to explain **why** the Phase 1 constraints arise, by isolating the host-visible mechanisms behind GPU interaction and identifying the abstraction pressure this creates for the OS. This is a mechanism-isolation phase, not a solution phase.

### 1.2 Current branch status

| Branch                      |         Experimental status |   Synthesis status | Notes                                   |
| --------------------------- | --------------------------: | -----------------: | --------------------------------------- |
| Synchronization topology    |                    Complete | Consolidated below | Completed subsection                    |
| Submission topology         |                    Complete | Consolidated below | Completed subsection                    |
| Allocation lifetime         |                    Complete | Consolidated below | Completed subsection                    |
| Scheduler / wakeup analysis | Complete enough for Phase 2 | Consolidated below | One concurrency comparison still useful |
| Context lifetime            |                    Complete | Consolidated below | Completed subsection                    |
| Multi-process GPU use       |                    Complete | Consolidated below | Completed subsection                    |
| Real workload validation    |       Pending consolidation |  Not yet assembled | To complete separately                  |
| Final Phase 2 synthesis     |                 In progress |        In progress | This document                           |

### 1.3 Definition of done for Phase 2

By the end of Phase 2, the following should be explicit:

1. What host-visible mechanisms dominate GPU interaction.
2. Which effects come from submission, synchronization, memory lifetime, initialization, and multi-process sharing.
3. What Linux can observe.
4. What Linux cannot infer.
5. What semantic structure a Phase 3 abstraction should expose.

---

## 2. Branch synthesis sections

# 2.1 Synchronization topology

## Question

What changes in host-visible behaviour when synchronization is moved from every iteration to the end of execution?

## Experimental procedure

This branch studied synchronization topology as one of the core Week 1 mechanism-isolation experiments in Phase 2. The aim was to explain how GPU completion becomes visible to Linux when host-side synchronization structure is changed. In particular, the comparison was between three conceptual variants: synchronization after every iteration, synchronization once at the end of execution, and a no-sync baseline. The purpose was not simply to count syscalls, but to identify the host-visible waiting structure behind `cudaStreamSynchronize()` and how that structure changes with different execution topologies.

The workload was a repeated CUDA kernel launch loop using a compute-heavy SAXPY-style kernel. Two main implementations were examined. In the `sync_every_iteration` case, each loop iteration launched one kernel and then immediately executed `cudaStreamSynchronize(expStream)`. In the `sync_end` case, many launches were issued first and a single synchronization was performed only after the full submission sequence. To make synchronization windows visible and attributable in `strace`, syscall-visible markers were inserted with `write(2, "START_SYNC\n", ...)` and `write(2, "END_SYNC\n", ...)` around `cudaStreamSynchronize()`. Earlier marker attempts using timing logs and `getpid()` / `gettid()` were discarded because they were not unique enough to anchor a specific sync episode cleanly.

Tracing was performed primarily with per-thread `strace -ff -ttt -T`, so that each runtime thread could be examined independently. This was important because CUDA runtime behaviour was not confined to the caller thread; helper and background threads also participated in waiting and coordination. Analysis then proceeded by locating marked sync windows and searching all corresponding thread traces for overlapping `poll`, `futex`, `ioctl`, and event-related activity. This shifted the analysis from coarse “process summary” inspection to semantically anchored, per-thread mechanism attribution.

## Main observations

First, in the forced slow-path `sync_end` case, a clearly marked synchronization window of about **228 ms** was captured, from `1772718387.661734` to `1772718387.889847`. Within that window, the caller thread itself did not show one obvious long blocking syscall. Instead, a helper/runtime thread remained blocked in repeated `poll(..., timeout=100)` calls on NVIDIA-related file descriptors and eventfds across the sync interval. This established that, at least in this slow-path case, synchronization was not exposed simply as “the main thread blocks in one syscall,” but as a multi-thread runtime protocol with helper-thread kernel sleeping.

Second, another runtime thread showed repeated `FUTEX_WAIT_BITSET_PRIVATE|FUTEX_CLOCK_REALTIME` calls with approximately **500 ms** timeout-driven behaviour and `ETIMEDOUT` returns. Because this pattern continued independently of the marked sync interval and because wake calls frequently returned `0`, it was interpreted as likely background runtime housekeeping rather than the direct GPU-completion path for the marked synchronization. Separately, in other runs and contexts, genuine 1:1 `FUTEX_WAIT` / `FUTEX_WAKE` pairings on the same futex address were observed, confirming that the CUDA runtime does internally use futex-based thread coordination.

Third, the `sync_every_iteration` case looked very different. The marked synchronization durations clustered stably around **20–25 ms**, with upper values around **30–34 ms**, but for the majority of these marked sync windows the mechanism counter was **zero**: no overlapping `poll`, `futex`, `ioctl`, or `eventfd` activity appeared between `SYNC_START` and `SYNC_END`. Re-running the experiment reproduced the same latency band and the same lack of visible syscall activity, indicating that this was not noise. In other words, many expensive per-iteration synchronizations were host-visible in time, but not host-visible as an intervening syscall sequence.

## Host-visible signature

The branch therefore produced two distinct host-visible signatures.

In the **single end-of-work synchronization** case, a slow synchronization window is associated with **helper-thread kernel-visible waiting**, especially repeated `poll()` on GPU- and event-related file descriptors, plus separate internal runtime futex coordination. This is a scheduler-visible waiting structure, but it is distributed across runtime threads rather than concentrated in one obvious caller-thread syscall.

In the **per-iteration synchronization** case, synchronization manifests mainly as **repeated fixed-latency waits with minimal or no syscall-visible structure inside the marked window**. The host thread spends roughly the expected kernel-completion time waiting, but the waiting path often does not cross the user-kernel boundary in a way visible to `strace`. The OS therefore sees elapsed synchronization time without a correspondingly rich syscall trace.

## Mechanism interpretation

The main mechanism-level interpretation is that CUDA synchronization does not have one universal host-visible form. It depends on topology.

When synchronization is delayed until the end of a larger body of outstanding GPU work, the runtime enters a slower completion path in which helper/event threads block in kernel primitives such as `poll()` while the runtime coordinates completion internally. In this case, Linux can observe some of the waiting structure, but still only indirectly: it sees generic wait primitives rather than an explicit “GPU work for stream X completed here” event.

When synchronization is performed after every iteration, the runtime often appears to wait in a way that is largely internal to user space, or at least not exposed through repeated blocking syscalls within the marked interval. That means `cudaStreamSynchronize()` can be expensive without being richly syscall-visible. The implication is that syscall-level observability alone is insufficient to reconstruct GPU completion behaviour under all synchronization regimes.

## What is invariant

Across both topologies, the fundamental invariant is that synchronization cost is real and repeatable, but the OS does not observe GPU completion directly. In both cases, Linux sees only boundary-level effects: thread blocking, generic wait primitives, occasional driver interactions, and elapsed time. It never receives a first-class semantic signal that identifies GPU phase completion in a way the scheduler can interpret directly.

A second invariant is methodological: clean per-thread tracing plus explicit syscall-visible markers were necessary to make defensible claims. Without semantic bracketing of the sync window, long `ioctl`s, futexes, and helper-thread activity were easy to over-attribute.

## Caveats

This branch did not directly capture the exact device-side instant of GPU completion. The traces only reveal host-visible waiting structure. GPU completion itself may propagate through interrupts, driver internal paths, and runtime state changes that do not appear as one explicit syscall return at the point of interest. Therefore, the strongest claims here concern **host-visible waiting and coordination**, not the full internal driver/device path.

In addition, scheduler-level quantities originally planned for this branch, such as `perf sched timehist`, `wchan`, and direct wake-to-run latency, were not fully integrated into the final result. The branch therefore closes with a strong mechanism interpretation, but not yet a fully quantified scheduler memo.

## Branch conclusion

Changing synchronization topology materially changed how GPU completion pressure appeared at the host boundary. In the slow-path `sync_end` case, synchronization was exposed through helper-thread kernel-visible waiting, especially repeated `poll()` activity, together with separate futex-based runtime coordination. In contrast, `sync_every_iteration` produced repeated ~20–25 ms synchronization delays that were usually syscall-silent inside the marked window, showing that substantial synchronization cost can exist without an equally explicit syscall signature. Overall, this branch established that Linux can observe host-side waiting structure around GPU completion, but not the completion semantic itself, and that the visibility of that structure depends strongly on synchronization topology.

---

# 2.2 Submission topology

## Question

How does restructuring the same total work into many small submissions versus fewer/larger submissions affect host-visible behaviour?

## Experimental procedure

This branch examined submission topology as a Week 1 mechanism-isolation experiment within Phase 2. The purpose was to test whether host-visible pressure arises from **launch granularity itself**, rather than from data size or final synchronization alone.

The experiment compared three submission variants that held the overall workload family constant while changing only the number of launches used to express that work:

* **1 submission**
* **100 submissions**
* **10,000 submissions**

Conceptually these correspond to coarse dispatch, moderate fragmentation, and extreme fine-grained fragmentation.

Instrumentation combined syscall-visible submission markers in `strace` with CUDA API and GPU kernel timing from Nsight Systems. The `strace` portion was used to inspect whether marked submission intervals corresponded cleanly to identifiable syscall episodes. Nsight Systems was then used to separate three components that `strace` alone could not disentangle reliably: **CUDA launch API time**, **GPU kernel execution time**, and **final stream synchronization time**.

## Main observations

First, the `strace` results for the **10,000-submission** case showed a clear early cold/warm split. The marker output contained `SUB_START|SUB_END` pairs for all **10,000** submission intervals. The first interval was much longer than the rest, approximately **129.5 ms**, while later submission intervals were only on the order of **tens of microseconds**. However, inspection of the long first marked interval on the main thread did **not** reveal a dense block of visible syscalls inside the marked region itself. Instead, some `ioctl`s appeared just before the marker window, and a helper thread was active nearby with `poll()` on `eventfd` and `/dev/nvidia0`, plus occasional futex wakes. The disciplined interpretation was therefore that the long first submission interval was real, but it did not map cleanly onto one obvious caller-thread syscall block; it occurred in the presence of overlapping runtime/helper-thread activity.

Second, the Nsight rerun data showed that **steady-state launches in the fragmented cases are individually cheap but collectively expensive**. In the rerun, the **100-submission** case had a median `cudaLaunchKernel` time of **2.935 µs** and a total launch time of **5.849 ms**. The **10,000-submission** case had a median launch time of **11.150 µs** but a total launch time of **366.372 ms**. Thus, the dominant structural change with increasing submission count is not that each launch becomes large, but that many very small launch episodes accumulate into a substantial host-side dispatch cost.

Third, increasing submission fragmentation also fragmented device execution. In the rerun data, **1 submission** produced a single GPU kernel of **4.445 ms**. The **100-submission** case produced GPU kernels with **75.527 µs average** and **38.400 µs median** duration, while the **10,000-submission** case produced kernels with **18.924 µs average** and **12.928 µs median** duration. Submission granularity therefore did not only restructure host launch activity; it also partitioned the GPU’s work into many smaller execution units. Total GPU kernel time rose from **4.445 ms** in the 1-submission case to **189.239 ms** in the 10,000-submission case.

Fourth, final synchronization time also increased with submission count, but it did **not** dominate the overall result. In the rerun data, `cudaStreamSynchronize` took **6.682 ms** for 1 submission, **11.800 ms** for 100 submissions, and **26.831 ms** for 10,000 submissions. This confirms that more fragmented execution also increases end-of-stream waiting, but the main structural story is still the accumulation of host launch cost and tiny-kernel execution rather than a single runaway sync event.

Finally, one-time startup effects were consistently present but unstable in exact attribution. Across runs, `cudaStreamCreate` was a major one-time setup cost, roughly **100–180 ms**, making it one of the most stable findings in the branch. By contrast, the largest launch outlier shifted between runs: in the first Nsight pass, the dramatic launch outlier appeared in the **100-submission** case, whereas in the rerun it appeared in the **10,000-submission** case with a **194.839 ms** maximum launch time. The safest interpretation is that cold-start or profiling-path setup effects are real, but the exact API call or exact benchmark variant that absorbs the largest one-time outlier is not stable enough to treat as the central phenomenon.

## Host-visible signature

The clearest host-visible signature of this branch is **dispatch amplification under fine-grained submission**.

At the coarse end, one submission appears as one millisecond-scale launch and one millisecond-scale kernel. At the fragmented end, submission becomes a long sequence of microsecond-scale launch episodes, while the GPU work is broken into many tiny kernels. The OS and runtime therefore see a much denser stream of fine-grained boundary events, even though each individual launch remains cheap in steady state.

A secondary signature is the presence of **significant one-time setup overhead**, especially around stream creation and early runtime activity, but with unstable exact placement across early API calls.

## Mechanism interpretation

The mechanism interpretation is that submission topology changes both **how often the runtime must enter the launch path** and **how finely the GPU work is expressed**.

With coarse submission, the host issues one larger dispatch episode, the GPU executes one larger kernel, and the overall host-visible structure is correspondingly simple. With fragmented submission, the runtime repeatedly performs very cheap launch operations, but their accumulation becomes substantial at large counts. At the same time, the device work itself is partitioned into much smaller kernels, which increases total kernel bookkeeping and execution accumulation. The effect is therefore not merely “more syscalls” or “a slower sync,” but a structural shift from one logical dispatch episode to many small dispatch and execution episodes.

The `strace` evidence also suggests that host submission occurs in the presence of background CUDA/NVIDIA runtime coordination threads, including helper-thread `poll()` and futex activity. However, the branch does **not** prove a direct causal chain from any specific helper-thread wait to any one particular submission delay.

This is also where the Phase 3 bridge becomes clear. In the highly fragmented case, the runtime knows that the 10,000 launches belong to one larger logical computation phase, but the OS sees only a long stream of small launch and execution events. The missing semantic structure is therefore **job/phase grouping**, **batch structure**, or **coarse-grain dispatch grouping**, rather than merely faster raw launch code.

## What is invariant

Several things remained stable despite changing submission topology.

First, there was always a meaningful one-time runtime/setup cost near the beginning of execution, especially around `cudaStreamCreate`, even if the exact placement of the largest early outlier varied. Second, multi-submission steady-state launch medians remained at the **microsecond scale** rather than becoming millisecond-scale in the normal case. Third, the host never received a semantic indication that these launches belonged to one logical batch or phase; across all variants, Linux still saw only boundary-level activity rather than structured GPU work semantics.

## Caveats

This branch did not fully complete all of the optional follow-up work that could sharpen the submission story further. In particular, the most useful remaining items would have been: one Nsight timeline screenshot for each variant to visually confirm launch clustering and sync placement, one controlled cold-vs-warm repeated-run protocol in the same process/session, and one fused-versus-many comparison point at comparable total work.

The branch also did not establish a one-to-one mapping from a marked submission interval to a unique visible syscall block. That limitation is itself informative, but it means the strongest quantitative claims come from the Nsight API/kernel timing data rather than from syscall timing alone.

## Branch conclusion

Submission topology materially changes the host-visible structure of GPU execution. A coarse configuration appears as one larger launch and one larger kernel, whereas fragmented configurations produce many microsecond-scale launches and many much shorter kernels. Although each steady-state launch remains individually cheap, aggregate launch cost grows substantially with submission count, and device work is fragmented in parallel with host submission. One-time setup costs are also significant, especially around stream creation, but their exact placement across early API calls is unstable across runs. Overall, the robust Phase 2 result is not a single giant outlier launch, but the transition from coarse dispatch to accumulated fine-grained host submission and device execution, which directly exposes the lack of semantic grouping available to Linux at the GPU boundary.

---

# 2.3 Allocation lifetime

## Question

How does allocation strategy translate into host-visible VM behaviour?

## Experimental procedure

This branch isolated the **allocation lifetime** mechanism family within Phase 2. The point was not to change the useful computation, but to change the **lifetime discipline of the memory objects supporting that computation** while keeping the broad copy, kernel, and synchronization structure as stable as possible.

The key experimental control variable was simply **where `cudaMalloc` and `cudaFree` were placed in program structure**. Two clean variants were compared:

* **allocOnce**: pinned host buffers allocated once; device chunk buffers allocated once before the loop and freed once after it; each iteration reused the same device chunk buffers.
* **allocEvery**: pinned host buffers still allocated once, but device chunk buffers were allocated and freed **inside every loop iteration**.

Measurement combined `strace` and `nsys`. The syscall families were grouped by role:

* **control / driver path**: `ioctl`
* **memory / VM**: `mmap`, `munmap`, `mprotect`, `brk`
* **sync / blocking**: `futex`, `poll`

The `nsys` summaries were then used to confirm that the intended structural intervention had really occurred: that allocation calls changed dramatically between the two variants, while kernels, memcpys, and synchronization remained essentially constant.

## Main observations

First, the **allocOnce** baseline produced a strongly **phase-localised** pattern. At the level of markers, the run contained one `START_MALLOC` and one `END_MALLOC`, consistent with a run-scoped allocation regime. In the main trace file, the top-level syscall counts were:

* `mmap`: **110**
* `munmap`: **9**
* `mprotect`: **15**
* `ioctl`: **425**
* `futex`: **17**
* `brk`: **55**

The more important finding, however, came from inspecting windows rather than only totals. Allocation-related activity in `allocOnce` was concentrated into a **compact setup region**, then largely absent during steady-state, and then reappeared in a compact **teardown region**.

Second, the **allocEvery** variant produced the critical contrast. The markers behaved exactly as intended: `START_MALLOC` and `END_MALLOC` appeared **2000** times in total, consistent with **1000 iterations** and therefore roughly **1000 per-iteration allocation episodes**. The top-level syscall counts in the main trace file were:

* `mmap`: **110**
* `munmap`: **9**
* `mprotect`: **15**
* `ioctl`: **5419**
* `futex`: **17**
* `brk`: **55**

Relative to `allocOnce`, almost every family stayed flat except one. The decisive change was:

* `ioctl`: **425 → 5419**

while:

* `mmap`: **110 → 110**
* `munmap`: **9 → 9**
* `mprotect`: **15 → 15**
* `futex`: **17 → 17**
* `brk`: **55 → 55**

Third, the repeated-allocation region in `allocEvery` was overwhelmingly dominated by control-path traffic rather than visible VM remapping. From the first `START_MALLOC` onward, the counts were:

* `ioctl`: **5004**
* `mmap`: **2**
* `munmap`: **2**
* `brk`: **1**

That makes the repeated mechanism extremely clear: once the per-iteration allocation regime starts, the dominant host-visible effect is repeated **driver/UVM control activity**, not repeated visible host mapping churn.

Fourth, `nsys` confirmed the structural intervention almost ideally. For `allocOnce`, the CUDA API summary reported:

* `cudaMalloc`: **2**
* `cudaFree`: **2**
* `cudaMemcpyAsync`: **4000**
* `cudaLaunchKernel`: **1000**
* `cudaStreamSynchronize`: **1000**

For `allocEvery`, the summary reported:

* `cudaMalloc`: **2000**
* `cudaFree`: **2000**
* `cudaMemcpyAsync`: **4000**
* `cudaLaunchKernel`: **1000**
* `cudaStreamSynchronize`: **1001**

This is very strong control. The only major CUDA API family that changed was **allocation/freeing**, while the rest of the workload shape remained almost fixed.

## Host-visible signature

The host-visible signature of this branch is **control-path churn under short allocation lifetime**.

In the `allocOnce` baseline, allocation-related host activity appears as compact **setup and teardown bursts** containing `ioctl` plus a smaller amount of `mmap` / `munmap` / `brk`, with relatively quiet steady-state execution. In the `allocEvery` variant, that quiet structure is replaced by repeated **driver/UVM `ioctl` amplification** throughout the loop.

The notable negative result is equally important: repeated device allocation did **not** appear as a matching explosion in visible `mmap` / `munmap` / `mprotect` counts. So, in this environment, repeated allocation lifetime is much more legible as repeated control-plane traffic than as repeated explicit VM-remapping traffic.

## Mechanism interpretation

The mechanism-level interpretation is that, in this CUDA/UVM stack, shortening device allocation lifetime from run-scoped to iteration-scoped does **not** primarily leak to Linux as repeated host VM remapping. Instead, it leaks primarily through repeated **runtime/driver bookkeeping**, visible at the OS boundary as large `ioctl` amplification.

There **is** one-time setup and teardown VM activity, but repeated per-iteration device allocation does not generate a one-for-one rise in visible host mapping syscalls in this setup. The repeated burden is absorbed and exposed mainly through control traffic between runtime and driver.

That is a useful Phase 2 mechanism-mapping result because it sharpens the abstraction-pressure story. The OS does not observe “this is a persistent buffer” versus “this is an ephemeral buffer.” It only sees coarse residue: setup/teardown memory activity and repeated control-path churn. The missing semantic structure is therefore **allocation lifetime intent** or **persistent-versus-ephemeral resource class**, not merely raw allocation count.

## What is invariant

Several things remained stable across the two variants.

First, the broad workload shape was preserved: memcpy count, kernel count, and synchronization count were essentially constant between `allocOnce` and `allocEvery`. Second, host-visible VM totals such as `mmap`, `munmap`, `mprotect`, and `brk` remained unchanged at the top-level count level. Third, synchronization-related behaviour on the main trace, at least at this coarse level, did not materially change either: `futex` stayed at **17**.

A second invariant is methodological: allocation lifetime is expressed entirely through **call placement**. There is no special CUDA “persistent mode” or “ephemeral mode”; the lifetime discipline is created by whether `cudaMalloc` / `cudaFree` are placed outside the loop or inside it.

## Caveats

This branch should not overclaim exact internal driver semantics. The result shows that repeated allocation is strongly **associated with** repeated `ioctl` traffic, but it does not prove which specific driver opcode corresponds to which internal action. Likewise, it should not be claimed that there is “no VM involvement at all,” because the alloc-once baseline clearly shows setup and teardown VM activity.

Also, although page faults and VM-region growth were part of the intended measurement suite, this result set mainly characterises the **syscall-level control-path manifestation** of allocation lifetime.

## Branch conclusion

The allocation-lifetime intervention showed that shortening device allocation lifetime from run-scoped to iteration-scoped does not primarily manifest as repeated visible host VM remapping in this CUDA/UVM environment. Instead, the dominant changed signal is a large amplification of driver/runtime control-path traffic, visible as repeated `ioctl` activity, while totals for `mmap`, `munmap`, `mprotect`, `brk`, and coarse synchronization counts remain essentially unchanged. The alloc-once baseline exhibits compact setup and teardown bursts with a relatively quiet steady state, whereas alloc-every-iteration replaces that structure with repeated control-heavy activity throughout the loop. Overall, this is a clean Phase 2 mechanism-level result: allocation lifetime is only coarsely visible to Linux, and in this setup the main observable pressure is fine-grained control-path churn rather than semantically legible VM-lifetime information.

---

# 2.4 Scheduler / wakeup analysis

## Question

How does Linux scheduling behaviour reflect GPU completion and host wakeup structure under varying thread/core pressure?

## Experimental procedure

This branch was designed to explain how GPU execution structure appears at the Linux scheduler boundary. The relevant Phase 2 aim here is not simply to count syscalls, but to isolate **scheduler-visible roles** such as dispatch, waiting, synchronization, and bookkeeping, then ask how those roles move when the execution structure is changed.

The branch varied five axes.

First, it varied **launch granularity**:

* **large** = 1 kernel launch covering the full row range
* **small** = 32 kernel launches, each handling a smaller row chunk

Second, it varied **synchronization topology**:

* `per_iter`: synchronize each iteration
* `final`: defer synchronization until the end

Third, it varied **CPU pressure placement**:

* burners on **separate cores**
* burners on the **same cores** as the benchmark

Fourth, it varied **burner count**:

* 0 burners
* 2 burners
* 4 burners

Fifth, it began exploring **host-side concurrency**, including an early case with **4 benchmark threads pinned to 2 cores**.

The intended logic was to vary how GPU work is partitioned, how often completion is surfaced back to the host, and how much contention the CPU-resident completion machinery experiences, then observe whether the OS-visible stress lands on submission (`ioctl`), VM activity, waiting (`poll`), or wake/synchronization (`futex`).

## Main observations

The first major finding was that the **dispatch path is almost invariant** across most of the single-thread runs. In nearly all of them, `ioctl` count remained **421**, and total `ioctl` time stayed in a narrow band of roughly **0.075–0.165 s**. VM activity also remained broadly similar. This is important because, given that the `small` case uses **32 launches** and the `large` case uses **1 launch**, a naive expectation would be that the OS-visible difference would show up mainly as more driver submission traffic. It did not.

The second major finding was that the dominant variation appears instead in the **completion and synchronization path**, specifically `futex` and `poll`. In the single-thread baseline:

* `per_iter_large`: **45 futex**, **4.501 s futex time**; **127 poll**, **4.830 s poll time**
* `per_iter_small`: **141 futex**, **27.503 s futex time**; **288 poll**, **27.549 s poll time**

That is a very large gap. The strongest immediate mechanism claim from the baseline is therefore that breaking work into many small launches does not meaningfully amplify dispatch, but it massively amplifies host-visible completion and wake behaviour.

The third major finding was that **launch granularity matters more than explicit sync placement for the small case**. For the large case, changing from `per_iter_large` to `final_large` reduced wait/sync time substantially, from roughly **4.5–4.8 s** down to about **2.5–2.8 s**. But for the small case, the effect was minimal:

* `per_iter_small`: about **27.5 s** in `futex` / `poll`
* `final_small`: about **26.5–26.9 s**

So the small case is structurally dominated by completion-path amplification to the point that deferring the explicit sync barely helps.

The fourth major finding was that **generic background CPU load is not the issue**. When two burners were placed on **separate cores**, the scheduler-visible behaviour barely changed:

* `per_iter_large` remained around **4.5 s**
* `per_iter_small` remained around **28 s**

The fifth major finding was that **same-core contention changes everything**. When burners were moved onto the **same cores** as the benchmark, the completion path inflated dramatically.

With **2 same-core burners**:

* `per_iter_large`: **130 futex**, **27.009 s futex time**; **285 poll**, **27.368 s poll time**
* `per_iter_small`: **332 futex**, **76.053 s futex time**; **767 poll**, **75.780 s poll time**

With **4 same-core burners** in `per_iter_small`:

* **434 futex**, **101.607 s futex time**
* **1020 poll**, **101.260 s poll time**

This shows two things very clearly. First, the host-side runtime/completion machinery is extremely sensitive to **direct CPU contention** on the cores where it runs. Second, the fine-grained 32-launch case is much more sensitive than the single-launch case.

The sixth major finding was that this effect **scales with contention** rather than simply appearing or not appearing. Going from 2 to 4 same-core burners in the `small per_iter` case increased the completion-side blow-up further, from about **76 s** to about **102 s** for both `futex` and `poll` time.

Finally, the early concurrency case suggested a **qualitatively different regime** can appear under oversubscription. In the early **4-thread / 2-core / small / per_iter** run, the summariser reported:

* **12,977,672 futex calls**
* **1482.173 s futex time**
* **3424 poll calls**
* **358.030 s poll time**
* **449 ioctl**, **0.796 s ioctl time**

This looks like the system entered a new regime dominated by synchronization churn and wake/sleep storms, where host-side coordination itself became the bottleneck.

## Host-visible signature

The host-visible signature of this branch is **completion-path amplification under fine-grained launches and direct CPU contention**.

Across the single-thread cases, the strongest movement is not in `ioctl`, but in `futex` and `poll`. The `small` launch-granularity case consistently amplifies those completion-side mechanisms. Deferring synchronization helps only for the `large` case. Separate-core load barely changes the result, but same-core contention causes a very large increase in wait/sync time. Under concurrency and oversubscription, the branch suggests a further transition into a **futex-storm regime**.

## Mechanism interpretation

The core mechanism interpretation is that **GPU progress is being surfaced to Linux through CPU-resident completion machinery whose responsiveness matters greatly**, and that this machinery is much more stressed by fine-grained launch structure than by coarse launch structure.

The near-invariance of `ioctl` means the launch-granularity change is not primarily being expressed as more visible driver submission. Instead, the major effect lands in the completion path. That is a strong abstraction-boundary result: Linux does not see launch restructuring mainly as more GPU work submission; it sees it mainly as more generic waiting and waking.

The small-launch case then shows that explicit synchronization placement is not the whole story. If the 32-launch case remains almost as bad under `final` as under `per_iter`, that means the cost is structurally tied to the fine-grained completion surface itself, not just to explicit host sync points in source code.

The burner experiments sharpen this further. Separate-core burners show that generic machine load is not enough to explain the effect. Same-core burners show that the host-side completion path depends critically on timely scheduling of the CPU threads responsible for runtime coordination, waiting, and wake handling. When those threads are directly contended, completion-side behaviour stretches dramatically, especially in the fine-grained case.

The early 4-thread / 2-core result then suggests a second-order effect: once fine-grained launch structure is combined with concurrency and oversubscription, the system can enter a qualitatively different synchronization-dominated regime, where host-side coordination overwhelms everything else. In that regime, the dominant visible symptom is a futex storm rather than increased submission traffic.

This all supports a broader Phase 2 claim. The OS does not see GPU work as a semantically grouped unit. It sees it mainly as a collection of generic wait/wake episodes on host CPU threads. Fine-grained GPU launch structure and CPU contention both amplify this visibility problem because GPU progress becomes dependent on timely scheduling of those CPU-resident completion threads.

## Final comparison still needed

The most useful remaining scheduler comparison is:

* **4 threads, `per_iter_large`, pinned to `0-1`**
* **4 threads, `per_iter_small`, pinned to `0-1`**
* **4 threads, `per_iter_small`, pinned to `0-3`**

Those three runs would separate whether the concurrency blow-up is specifically tied to fine-grained launch structure rather than simply to “more threads,” and whether **oversubscription itself** is a key ingredient.

## What is invariant

Several things remain stable across the explored cases.

First, the **dispatch path** is almost invariant in the single-thread runs: `ioctl` count remains essentially fixed. Second, generic background CPU load on other cores does not materially change the regime. Third, the dominant visible pressure continues to be generic host blocking/wake behaviour rather than semantically rich information about GPU execution phases.

## Caveats

The scheduler branch should not overclaim exact internal runtime or driver semantics. It shows where the host-visible stress lands, but not the full device-side arbitration path.

Also, the early high-concurrency summariser output became messy because the trace volume was so large. The combined totals are still clear enough to support the “new regime” interpretation, but those runs would benefit from reduced iteration counts and cleaner marker richness in future passes.

Finally, this branch is strongest on `strace`-visible scheduler roles. The original Phase 2 plan also expected deeper `perf sched` and runqueue analysis, and that remains a useful optional complement when tooling is available again.

## Branch conclusion

The scheduler exploration showed that the main OS-visible effect of restructuring GPU work is not a large change in driver submission, but a large change in host-side completion and synchronization behaviour. Fine-grained launch structure amplifies `futex` and `poll` dramatically, while deferring synchronization helps only for the single-launch case. Generic background CPU load on separate cores has little effect, but direct same-core contention causes major blow-ups in wait/sync time, especially for the fine-grained case. Under concurrency and oversubscription, the system can enter a synchronization-dominated futex-storm regime. Overall, this branch supports a strong Phase 2 claim: Linux mainly experiences GPU progress through CPU-resident wait/wake machinery rather than through a semantically grouped execution abstraction, and that machinery becomes a major bottleneck under fine-grained launch structure and host CPU contention.

---

# 2.5 Context lifetime

## Question

What is the host-visible initialization signature of GPU context creation, and what is one-time versus repeated per-process setup cost?

## Cases to recover

This branch considered the following context-lifetime cases:

* **single-process first-use / directed launch**
* **warm versus cold behaviour where available**
* **two-process staggered creation**
* **two-process simultaneous creation**

## Experimental procedure

This branch was not about steady-state kernel behaviour. It was about the **host-visible cost and structure of bringing CUDA execution into existence**. The core questions were: what GPU runtime and context initialization look like from Linux; which components are machine-wide or amortized versus repeated per process; how this behaviour differs across cold, warm, staggered, and simultaneous cases; and whether multi-process bring-up exposes serialization or amplification effects that simpler single-process traces hide.

The general method was to instrument CUDA microbenchmarks with explicit phase markers, including labels such as `PROGRAM_START`, `BEFORE_STREAM_CREATE`, `AFTER_STREAM_CREATE`, `BEFORE_HOST_ALLOC`, `AFTER_HOST_ALLOC`, `BEFORE_DEVICE_ALLOC`, `AFTER_DEVICE_ALLOC`, `BEFORE_KERNEL1`, `AFTER_KERNEL1_LAUNCH`, `BEFORE_SYNC1`, `AFTER_SYNC1`, and `PROGRAM_END`. Per-thread `strace -ff -ttt -T` traces were then collected and inspected for `mmap`, `munmap`, `mprotect`, `clone3`, `futex`, `ioctl`, and occasional writes to internal runtime/driver file descriptors. These traces were compared across single-process and multi-process cases, with staggered and simultaneous launch structures used to separate per-process repetition from concurrency-induced overlap.

## Main observations

First, a recurring **initialization signature** appeared across the branch before the first meaningful kernel execution. Very early in each process, Linux observed a large burst of loader- and runtime-related mapping activity, including many `mmap`, `mprotect`, and `munmap` calls. This is best interpreted as userspace and library bring-up rather than GPU work in the narrow sense, but it is still part of the observable startup path required before any real CUDA interaction can occur.

Second, very early after initial setup, helper threads were created via `clone3`. This is a critical structural observation: GPU use is not just a main-thread sequence of direct driver calls. The CUDA runtime establishes a small internal control plane of helper threads before usable execution begins.

Third, after early runtime activation, traces showed a dense burst of `ioctl` activity, often mixed with additional mappings, including shared mappings, repeated small mappings, and fixed-address mappings. This is the clearest host-visible boundary where the userspace runtime begins negotiating real GPU-related state with the kernel driver.

Fourth, startup also established waiting and signalling infrastructure before the first useful GPU work. Across traces there were futex waits and wakes, small writes to internal file descriptors, and recurring wait- or poll-like driver interactions on certain file descriptors. That suggests startup is not only about allocation and registration of control structures, but also about building the machinery used later for completion tracking and coordination.

Taken together, these recurring elements formed a clear branch-wide pattern: loader and mapping activity, helper-thread creation, dense driver negotiation, additional GPU-related mappings, and early coordination infrastructure all appear before the first meaningful execution episode.

## Host-visible signature

The host-visible signature of context lifetime is a **substantial, structured startup episode** rather than a tiny invisible prelude.

From Linux’s point of view, first-use CUDA bring-up appears as a compound pattern containing:

* large early `mmap` / `mprotect` / `munmap` bursts
* helper-thread creation via `clone3`
* dense `ioctl` negotiation with the NVIDIA driver
* additional shared and fixed-address mappings
* futex and signalling setup for later coordination

This is important because it means the OS can clearly observe that a process is undergoing a major startup phase, but not what that phase semantically means unless the runtime phase boundaries are already known.

## Mechanism interpretation

The main mechanism-level interpretation is that **context/runtime bring-up is structurally rich, repeated, and only coarsely legible to Linux**.

In the single-process case, some startup costs could be absorbed into stream creation, first launch, first synchronization, or first allocation, rather than surfacing as one clean standalone “initialization block.” That does **not** mean the startup work is absent. It means that in simple runs the work can be folded into a smaller number of visible host phases.

The staggered two-process case then clarifies the per-process structure. Because the processes are offset in time, each process shows a similar repeated startup signature: loader mappings, helper-thread creation, dense initialization `ioctl`s, shared/fixed mappings, then only later stream creation, allocation, first kernel, synchronization, and cleanup. This is strong evidence that the dominant visible initialization burden is **not purely machine-wide and one-shot**. A substantial portion clearly repeats per process.

The simultaneous two-process case is where the branch becomes most revealing. When both processes initialize at nearly the same time, their startup episodes overlap and become noisier and more interleaved. That suggests some degree of shared runtime/driver contention, internal serialization, or at least resource competition on host-side control paths. The result does **not** prove full global serialization, but it does show that concurrent context/runtime creation is not free and not perfectly isolated.

## Open questions

A JIT- or setup-looking anomaly appeared more clearly in one two-launch structure than in some simpler directed single-launch cold/warm cases. The best interpretation is that lazy setup surfaced at different semantic boundaries, or that some execution structures made the startup episode easier to expose in `strace`. This is interesting, but it does not undermine the main model.

## Branch conclusion

GPU context/runtime bring-up has a clear, structured host-visible signature. Linux sees it as a substantial startup episode consisting of loader and mapping activity, helper-thread creation, dense driver `ioctl` negotiation, additional shared/fixed mappings, and early wait/signalling setup before the first meaningful kernel execution. The staggered two-process case shows that much of this burden repeats per process, while the simultaneous case shows that concurrent bring-up amplifies visible complexity and suggests contention or partial serialization effects. Overall, this branch establishes that Linux can observe initialization pressure but not its semantics: it can tell that substantial startup work is happening, but not cleanly distinguish context establishment, runtime setup, stream/control-plane setup, and launch preparation without additional semantic structure.

---

# 2.6 Multi-process GPU use

## Question

What changes when GPU use is shared across processes rather than threads, and how does MPS change the host-visible structure?

## Completed experiment

This branch was designed as the focused **multi-process GPU-use study** that still remained distinct from the context-lifetime work. Its purpose was to move beyond single-process and multi-threaded cases and study what Linux can observe when **two independent processes share the same GPU**, both **without MPS** and **with MPS**. The intended observables were host-visible fairness effects, interference, completion/wait structure, and any evidence of process-level serialization or asymmetric service.

A deliberately small and interpretable two-process benchmark was used. Across the four runs, two asymmetry axes were varied:

* **submission topology asymmetry**

  * one process used **micro** submission granularity
  * the other used **large** submission granularity
  * matrix size otherwise matched

* **work-size asymmetry**

  * one process used **2048 × 2048**
  * the other used **512 × 512**
  * submission style was otherwise comparable

Each family was then run in two GPU-sharing modes:

* **without MPS**
* **with MPS**

For each run, the same summary procedure measured per-process iteration-duration distributions, per-process attributed wait-time distributions, per-process attributed `ioctl`-time distributions, cross-process iteration skew, and cumulative drift between the two processes over time.

## What to compare

This branch compared the following dimensions:

* **futex / poll / wait structure**
* **host-visible asymmetry in iteration completion**
* **process-level skew and cumulative lag**
* **MPS versus no-MPS sharing regime**
* **submission-topology asymmetry versus plain workload-size asymmetry**

The last comparison turned out to be the central result of the branch.

## Main observations

Across the four runs, the severity ordering from most asymmetric to least asymmetric was:

1. **micro vs large submission topology, no MPS**
2. **micro vs large submission topology, MPS**
3. **2048 vs 512 matrix size, no MPS**
4. **2048 vs 512 matrix size, MPS**

This is already a strong finding. It shows that the strongest interference lever here is not merely “more work per iteration,” but **how work is structured and submitted**.

In the **no-MPS micro-vs-large topology** case, the asymmetry was extreme. The slower process had a **106.330 ms** p50 iteration duration, while the faster process had **1.319 ms**, for a p50 absolute skew of **104.125 ms** and a mean absolute skew of **102.961 ms**. Worst skew reached **482.150 ms**, and by iteration 499 the cumulative time gap had grown to more than **51 s**. The slow process was dominated by host-visible waiting, with **111.414 ms** p50 wait, **617.352 ms** p95 wait, and **635.242 ms** p99 wait, while the fast process showed essentially zero wait in the common case.

In the **no-MPS 2048-vs-512 size** case, asymmetry was still present but looked much more regular and proportional. The slower side had **33.232 ms** p50 iteration duration, the faster side **6.774 ms**, and the p50 absolute skew was **26.178 ms**, with **27.945 ms** mean absolute skew and **53.179 ms** worst skew. The final cumulative gap was **13.973 s**. The slower process again showed substantial wait, with **33.755 ms** p50 wait and long wait tails, but the overall story was much more intuitive: the larger workload took longer.

In the **MPS micro-vs-large topology** case, MPS clearly improved the absolute regime, but did not eliminate the asymmetry. The slower side improved from **106.330 ms** p50 to **61.100 ms**, and p50 skew dropped from **104.125 ms** to **60.602 ms**. Mean absolute skew was still **61.975 ms**, worst skew **78.805 ms**, and cumulative drift by the end of the run was still about **30.988 s**. The slower process continued to show large waits, with **100.113 ms** p50 wait and p95/p99 waits near **600 ms**, while the faster process remained almost perfectly clean.

In the **MPS 2048-vs-512 size** case, the regime was the calmest and most proportional of all four runs. The slower side had **20.590 ms** p50 iteration duration, the faster side **4.017 ms**, and p50 absolute skew was **16.557 ms**, with **17.177 ms** mean absolute skew and **20.345 ms** worst skew. Final cumulative drift was **8.588 s**. The faster process had effectively zero p50 and p95 wait, and the slower process had zero p50 wait with only tail spikes.

## Mechanism interpretation

The main mechanism-level interpretation is that **shared GPU use across processes is more sensitive to submission structure than to raw workload size**.

Changing matrix size produced moderate, regular, mostly intuitive skew. One process simply had a larger service demand per iteration. Changing submission topology, by contrast, produced much larger and more pathological asymmetry, including persistently elevated waits for one process and near-zero waits for the other. That means the system is more sensitive to **how work is packaged and exposed to the runtime** than merely to **how much work exists**.

A second key interpretation is that **MPS changes the regime but does not solve the topology problem**. MPS improved absolute times, reduced skew, and reduced cumulative drift in both experiment families. But the degree of improvement was much stronger for plain size asymmetry than for topology asymmetry. So MPS is helpful, but it is not a sufficient semantic fairness abstraction for topology-driven interference.

A third important interpretation is that the dominant host-visible symptom of multi-process interference is **generic waiting**, not semantically rich GPU information. Across all four runs, the strongest recurring signal was one process exhibiting elevated per-iteration wait behaviour, while the attributed `ioctl`-time view remained negligible in the per-iteration summaries. That does **not** mean submission path costs are irrelevant; it means that in this analysis view, the main host-visible symptom of interference is blocking and delay, not a directly interpretable explanation of why the GPU service discipline became asymmetric.

## Branch conclusion

The focused multi-process GPU-use study showed that shared GPU execution across processes is not governed solely by how much work each process submits. Instead, it is highly sensitive to **how that work is structured**. Raw matrix-size asymmetry produced moderate, mostly proportional skew, whereas submission-topology asymmetry produced much larger and more pathological process-level imbalance. MPS improved absolute behaviour and reduced skew in both cases, but did not eliminate topology-driven asymmetric service. Across all runs, the main host-visible symptom of interference was generic wait behaviour rather than semantically meaningful device-side information. Overall, this branch strengthens the Phase 2 claim that the current OS/GPU boundary lacks the semantic grouping needed to reason about shared GPU work, and that higher-level work-unit or phase structure matters for interference in ways Linux cannot currently infer.

---

# 2.7 Real workload validation (minimal serving stack)

## Question

Which synthetic signatures reappear in a real serving stack, and which branch best explains the observed host-visible behaviour?

## Current status

This section is not yet consolidated in this document.

## Minimal setup used

* [To fill]

## What to compare against

* submission topology
* synchronization topology
* context lifetime
* allocation lifetime
* scheduler branch

## Main observations

* [To fill]

## Mechanism interpretation

* [To fill]

## Optional re-check question

Does one additional clean serving trace materially sharpen the correspondence claim?

## Branch conclusion

* [To fill]

---

## 3. Mechanism atlas

| Phenomenon                     | Host-visible signature                                                                                                  | Mechanism interpretation                                                    | What Linux can infer                             | What Linux cannot infer                                | Phase 3 implication                                 |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------ | --------------------------------------------------- |
| Fine-grained submission        | Dispatch amplification and fine-grained work fragmentation under high submission counts                                 | Many individually cheap launches accumulate; device work is also fragmented | That host boundary activity is much denser       | Which launches belong to one logical computation phase | Coarse-grain dispatch grouping / phase grouping     |
| Frequent synchronization       | Helper-thread kernel-visible waiting in some regimes; syscall-silent delay in others                                    | Sync path depends on dynamic runtime state and topology                     | That host-side waiting is happening              | Exact GPU completion semantics                         | Phase-boundary / completion grouping hints          |
| Allocation churn               | Short allocation lifetime appears primarily as driver/UVM control-path churn rather than repeated visible VM remapping  | Repeated allocation leaks mainly through control-plane bookkeeping          | That control-path pressure rises sharply         | Persistent vs ephemeral allocation intent              | Lifetime-class hints / resource ownership semantics |
| Context initialization         | Dense multi-thread startup episode of mappings, helper-thread creation, and driver negotiation                          | Context/runtime first use is substantial and repeats per process            | That startup pressure exists                     | Which semantic startup phase is being performed        | Initialization / setup phase grouping               |
| Thread-level contention        | Fine-grained launches amplify `futex`/`poll`; same-core contention and oversubscription make it much worse              | GPU progress reaches Linux through CPU-resident completion machinery        | That completion-side wake/sleep pressure is high | Device-side arbitration or grouping                    | Host-side phase/cohort grouping around completion   |
| Process-level sharing          | Shared GPU use appears mainly as asymmetric wait-heavy service; topology produces stronger skew than raw size imbalance | Process interference is highly sensitive to submission structure            | That one process is repeatedly delayed           | Why that process lost service                          | Higher-level work-unit and granularity hints        |
| Real serving request execution | [To fill]                                                                                                               | [To fill]                                                                   | [To fill]                                        | [To fill]                                              | [To fill]                                           |

---

## 4. Defensible Phase 2 claims

### Claim 1

**The dominant OS-visible effect of many GPU restructuring interventions is not increased submission traffic but increased generic host-side waiting and coordination pressure.**

**Evidence:**

* Scheduler branch: `ioctl` remained nearly constant while `futex`/`poll` changed dramatically.
* Synchronization branch: costly per-iteration waits were often syscall-silent, while slow-path end-sync exposed helper-thread waiting.
* Multi-process branch: per-iteration interference appeared mainly as wait-heavy asymmetry rather than explanatory `ioctl` time.

**Scope:**

Applies to the current CUDA/NVIDIA/UVM host-visible boundary in these microbenchmarks.

**Limitation:**

Does not reveal device-side arbitration or exact internal runtime semantics.

### Claim 2

**Fine-grained launch and submission structure is a stronger source of host-visible pressure and interference than raw work size alone.**

**Evidence:**

* Submission branch: 10,000 tiny launches caused large aggregate launch time and heavy kernel fragmentation.
* Scheduler branch: 32-launch structure amplified completion/wake behaviour far more than 1-launch structure.
* Multi-process branch: topology asymmetry caused much worse process skew than simple 2048-vs-512 size imbalance.

**Scope:**

Applies to the explored microbenchmark families and process-sharing regimes.

**Limitation:**

Not yet validated against the serving stack in this document.

### Claim 3

**Allocation lifetime and context lifetime are only coarsely legible to Linux; the OS sees their residue, not their semantics.**

**Evidence:**

* Allocation branch: repeated device allocation amplified `ioctl`, not visible `mmap`/`munmap` totals.
* Context branch: first-use setup appeared as mappings, helper-thread creation, and driver negotiation, but not as one clean semantic event.

**Scope:**

Applies to this CUDA/UVM environment.

**Limitation:**

Full page-fault/VMA analysis is still incomplete.

### Claim 4

**Host CPU contention materially distorts the OS-visible completion path for GPU work, especially when execution is fine-grained.**

**Evidence:**

* Scheduler branch: same-core burners caused major increases in `futex`/`poll` time; separate-core burners did not.
* Early oversubscribed concurrency run entered a futex-storm regime.

**Scope:**

Applies to host-side completion machinery under direct core contention.

**Limitation:**

Still needs one cleaner final comparison to isolate oversubscription versus launch granularity under 4-thread runs.

### Claim 5

**The common abstraction failure across Phase 2 is missing semantic grouping of GPU work.**

**Evidence:**

* Submission: Linux sees many launches, not one logical batch.
* Synchronization: Linux sees waiting, not explicit completion semantics.
* Allocation/context: Linux sees setup and control-plane residue, not lifetime class or phase.
* Multi-process sharing: Linux sees asymmetric delay, not the work-unit structure that caused it.

**Scope:**

This is the broadest cross-branch claim and the main bridge to Phase 3.

**Limitation:**

Needs real-stack validation to confirm how directly the same abstraction failure transfers to serving behaviour.

---

## 5. Phase 3 bridge

## 5.1 Missing semantic structure

The recurring limitation suggested by Phase 2 is that Linux can observe boundary activity, but cannot natively recover the semantic grouping of GPU work. It sees generic submission, waiting, allocation/setup residue, and blocking, but not the higher-level structure that gives those events meaning.

## 5.2 Candidate abstraction direction

Candidate Phase 3 directions suggested by the current synthesis are:

* **phase-boundary hints** for initialization, allocation/setup, dispatch, and completion
* **coarse-grain dispatch grouping** so many launches can be exposed as one logical work unit
* **batch / job structure visibility** for GPU work that is currently only visible as repeated small submissions
* **lifetime-class hints** for persistent versus ephemeral resources
* **submission-granularity or execution-phase hints** for shared-process interference analysis

## 5.3 Requirements imposed by Phase 2

A Phase 3 abstraction should:

* preserve low overhead;
* expose grouping/phase semantics absent from raw syscall traces;
* help distinguish submission, waiting, initialization, and memory-lifetime effects;
* remain meaningful under concurrency and multi-process sharing;
* improve attribution and reasoning without requiring kernel or framework rewrites.

## 5.4 What not to overclaim

Phase 2 does **not** show direct device-side scheduling semantics. It shows host-visible residue and structural pressure at the OS boundary. The Phase 3 prototype should therefore be framed as a minimal semantic export layer for better grouping and attribution, not as a claim to fully solve GPU scheduling fairness or eliminate all runtime overhead.

---

## 6. Tutor meeting preparation

## Completed in Phase 2

* synchronization topology synthesis
* submission topology synthesis
* allocation lifetime synthesis
* scheduler / wakeup synthesis
* context lifetime synthesis
* focused multi-process GPU-use synthesis

## Still being closed

* real serving-stack validation
* final integrated correspondence to a real workload
* one final scheduler concurrency comparison if desired
* final one-page meeting note

## Questions for the tutor

1. Is the current Phase 2 closure sufficient if the remaining work is tightly scoped to serving validation and final synthesis?
2. Which Phase 3 abstraction direction seems most justified by the current evidence: phase-boundary hints, dispatch grouping, lifetime classes, or a combination?
3. Is the missing final scheduler comparison worth doing before Phase 3, or is the current mechanism picture already strong enough?

---

## 7. Immediate working checklist

### Synthesis first

* [x] Pull prior branch syntheses into Sections 2.1–2.6.
* [x] Convert each completed branch into: question, procedure, observations, signature, interpretation, conclusion.
* [x] Fill an initial mechanism atlas.

### Analysis and consolidation only

* [ ] Rebuild serving-validation subsection from existing material.
* [ ] Perform serving-stack re-check only if it sharpens correspondence or confidence.
* [ ] Decide whether to run the final scheduler concurrency comparison.

### Final outputs

* [x] Write 3–5 defensible claims.
* [x] Draft the Phase 3 bridge.
* [ ] Prepare one-page tutor meeting note.
* [ ] Extract the semantic requirements the Phase 3 abstraction must satisfy in final form.

