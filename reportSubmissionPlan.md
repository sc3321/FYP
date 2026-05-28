# Final Report + Implementation Plan (28 May – 12 June)

**Deadline:** Report due Thursday 12 June. Post-submission window 13–22 June for extensions, presentation prep, viva polish.

**Two parallel tracks every day:** report writing + implementation. Neither lands without the other.

---

## The compass — return here whenever the story slips

Semantic structure — which host activity forms one logical unit of GPU work, and whether it
is latency-critical — is known only at the **application level**. As work descends the stack it
is progressively stripped: the runtime retains a partial structural shadow (streams, launch
order, context); by the OS boundary the semantics are gone, leaving only generic
`ioctl`/`futex`/`poll`/`mmap` activity. This is **not a measurement problem** — Phase 2 shows
no structural lever produces an OS-visible signal that recovers the lost semantics. Closing the
gap requires a **channel** that lets application-level semantics reach the coordination layer.
This project designs the **minimal** such channel and demonstrates one policy that consumes it.

Every paragraph must do one of four things:
1. Establish the gap exists (Background, Mechanism Characterisation)
2. Design the minimal closure (Abstraction Design)
3. Demonstrate the closure works (Implementation, Evaluation)
4. Reflect on what it enables (Discussion)

If a paragraph does none of these, cut it.

---

## Locked report structure

| Ch | Title | Target pages |
|----|-------|--------------|
| — | Abstract | 1 |
| 1 | Introduction | ~4 |
| 2 | Background (GPUs, OS, visibility boundary) | ~9–10 |
| 3 | Related Work (critical review) | ~5 |
| 4 | Mechanism Characterisation (Phase 1 + 2) | ~12 |
| 5 | Abstraction Design | ~6 |
| 6 | Implementation | ~7 |
| 7 | Evaluation | ~10 |
| 8 | Discussion | ~3 |
| 9 | Conclusion and Future Work | ~2 |
| — | References / Declarations / Appendices | not counted |

**Hard cap: 60 content pages, 11pt, 2.5cm margins. 40 tight pages beats 60 padded ones.**
Source code lives in the repo, NOT the report.

**Contribution stance: A** — the abstraction is the contribution; the policy is one demonstration.
Goals are NARROW (only what the prototype does). Breadth (DVFS, affinity, attribution) lives in
Discussion as *implication*, never as a goal. Goals create evaluation obligations; implications
do not.

---

## Implementation status

**Must-have by 12 June**
1. llama.cpp decode-level instrumentation + final results matrix (10–20 runs averaged)
2. eBPF host-side evidence (4 selected runs)
3. LD_PRELOAD sync wrapper working on ≥1 program

**Nice-to-have / post-submission (13–22 June)**
4. Ring buffer
5. Policy daemon polish / live snapshots
6. CPU affinity experiments (NEW experiment — do NOT start before 12 June)
7. Extended LD_PRELOAD intercepts (launch + sync)

**Also remember**
- Repo in submission-ready shape: clean README, build + run + plot-regen instructions
- Plot scripts as one-command reproducible: lc_tail_microbench, be_throughput_microbench,
  llama_policy, ebpf_summary
- 10–20 averaged runs is REAL work (single-run p99s are unreliable) — budget wallclock

---

## Day by day

### Thu 28 May — Chapter 5 (Abstraction Design) [report-priority]
- AM (4h, report): §5.1 Goals & non-goals, §5.2 The phase abstraction — drafted together
- PM (3h, report): §5.3 classification, §5.4 architecture, §5.5 policy-as-consumer
- Eve (2h, impl): start llama.cpp decode instrumentation (code only, no runs)
- **Deliverable:** Chapter 5 first draft (~6pp) + 1-paragraph stub atop every other chapter

### Fri 29 May — Chapter 6 + llama.cpp instrumentation
- AM (4h, impl): finish llama.cpp decode instrumentation (critical — results depend on it)
- PM (4h, report): Chapter 6 §6.1–6.3 (phase library/API, shared-mem + robust locking, policy)
- Eve (1h, report): §6.4 event logging + daemon
- **Deliverable:** decode instrumentation done; Chapter 6 ~60% drafted

### Sat 30 May — implementation-priority
- AM (3h, impl): kick off long llama.cpp matrix (10–20 runs/case, runs unattended)
- PM (4h, report): Chapter 6 finish (§6.5 LD_PRELOAD, §6.6 build/deploy); split Background
  into new Ch2 + Ch3 skeleton
- Eve (2h, impl): check run progress, queue reruns
- **Deliverable:** results runs in flight; Chapter 6 drafted; Ch2/Ch3 skeleton exists

### Sun 31 May — eBPF setup + Discussion outline
- AM (4h, impl): build 4 eBPF scripts (off-CPU, sched-switch, syscall latency, run wrapper)
- PM (3h, report): Chapter 8 outline + §8.5 (what other policies the abstraction enables — DVFS)
- Eve (2h, impl): run eBPF on LC-alone baseline, sanity-check output
- **Deliverable:** eBPF tooling works; Discussion outline + DVFS section drafted

### Mon 2 Jun — Chapter 4 (Mechanism Characterisation)
- Full day (8h, report): Ch4 (Phase 1 + Phase 2 + mechanism atlas). Weave in: *every branch
  is an instance of the semantic gap*
- Eve (1h, impl): check llama.cpp matrix progress
- **Deliverable:** Chapter 4 drafted (~12pp)

### Tue 3 Jun — eBPF runs + LD_PRELOAD start
- AM (4h, impl): eBPF on 4 cases (LC-alone, LC+BE-long no-policy, LC+BE-long policy,
  LC+BE-chunked no-policy)
- PM (3h, impl): LD_PRELOAD wrapper — cudaStreamSynchronize + cudaDeviceSynchronize intercepts
- Eve (2h, report): Ch2 polish — rewrite synthesis to point at the gap + Phase 2
- **Deliverable:** eBPF data collected; LD_PRELOAD core working

### Wed 4 Jun — LD_PRELOAD finish + Related Work
- AM (3h, impl): finish + test LD_PRELOAD (simple CUDA prog + ideally one llama.cpp run)
- PM (4h, report): Chapter 3 (Related Work — Paella, TGS, LithOS, Orion) critical review
- Eve (2h, report): consolidate ALL results; confirm Chapter 7 has everything it needs
- **DECISION POINT:** if impl solid, fully shift to report Thu onward. If LD_PRELOAD fragile,
  give it Thu AM.
- **Deliverable:** implementation must-haves done; Chapter 3 drafted

### Thu 5 Jun — Chapter 7 (Evaluation) start
- Full day (8h, report): §7.1 Methodology, §7.2 microbench results (with figures)
- Eve (1h): verify results clean + reproducible
- **Deliverable:** Chapter 7 ~40% drafted

### Fri 6 Jun — Evaluation continued
- AM (4h, report): §7.3 llama.cpp real-workload results (with figures)
- PM (3h, report): §7.4 eBPF results
- Eve (2h, report): §7.5 overheads, §7.6 evaluation limitations
- **Deliverable:** Chapter 7 drafted (~10pp)

### Sat 7 Jun — Discussion + Conclusion
- AM (4h, report): Chapter 8 fleshed from outline
- PM (3h, report): Chapter 9 (Conclusion + Future Work)
- Eve (2h, report): coherence read-through of Ch4–9
- **Deliverable:** Chapters 8 + 9 drafted

### Sun 8 Jun — Introduction + Abstract + repo
- AM (4h, report): Chapter 1 (Introduction) — written last, now it knows what it can promise
- PM (2h, report): Abstract (1 page, no citations)
- Late PM (3h, repo): README, build/run/plot-regen scripts, cleanup
- **Deliverable:** full draft complete end-to-end; repo submission-ready

### Mon 9 Jun — Cut day
- Full day (8h, report): first major edit pass. Find padding, remove it. Target ≤55pp with
  headroom. Be ruthless.
- **Deliverable:** tightened draft

### Tue 10 Jun — Citations + figures + formatting
- AM (4h): every claim cited; Vancouver format checked
- PM (3h): every figure captioned + referenced; numbering correct
- Eve (2h): Declarations (AI tool use, ethics, originality)
- **Deliverable:** reference-complete, figure-complete, declarations done

### Wed 11 Jun — Buffer
- For the unexpected. If fine: final cold read-through. If broken: fix.

### Thu 12 Jun — Submission
- AM: final proofread, PDF, submit to BOTH Scientia + Turnitin by lunch
- PM: rest

---

## Contingency — what gives if you slip (decide NOW)

1. First cut: extended LD_PRELOAD intercepts (sync-only is enough)
2. Second cut: Discussion depth (keep §8.5 DVFS; tighten rest to one para each)
3. Third cut: number of llama.cpp configs (8 cases is plenty)
4. **Never cut:** the four load-bearing results — microbench C/D, microbench E/F,
   llama.cpp C/D, eBPF summary

---

## Health rule
Sleep, eat, walk. The plan assumes full cognitive capacity. A burned-out day 16 is worse than
a focused day 14.
