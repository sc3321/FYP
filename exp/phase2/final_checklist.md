# Phase 2 Final 10-Day Plan
**Window:** Friday 10 April 2026 → Monday 20 April 2026  
**Target:** finish Phase 2 cleanly and enter Phase 3 with a clear prototype direction.

---

## Actual remaining work

### Already completed experimentally
These branches already have results and analyses distributed across earlier project chats:
- synchronization
- submission topology
- allocation lifetime
- scheduler deep dive

For these, the remaining work is **synthesis**, not new experimentation.

### Still left
- **Context lifetime re-analysis**
  - traces/results already exist
  - analysis and interpretation need to be rebuilt
- **One focused multi-process GPU-use study**
  - actual GPU use across processes
  - ideally **without MPS** vs **with MPS**
- **Minimal real serving-stack validation**
- **Phase 2 synthesis / write-up**
- **Supervisor meeting**
- **Optional perf decision**

---

## Priorities, tasks, and deliverables

### Priority 1 — Context lifetime re-analysis
This is the main incomplete Phase 2 branch.

**Tasks**
- Re-run the context-analysis scripts over the existing data.
- Recover the interpretation for:
  - single context creation
  - simultaneous multi-process context creation
  - staggered multi-process context creation
  - warm vs cold behavior if available
- Rebuild the lost tables and summaries.

**Questions**
- What is the host-visible initialization signature of GPU context creation?
- What is one-time setup vs repeated per-process overhead?
- Does simultaneous creation introduce serialization or amplification?
- How does staggered creation differ?

**Deliverable**
- Final **Context Lifetime** subsection
- Summary table of signatures and conclusions

---

### Priority 2 — Focused multi-process GPU-use study
This remains an actual missing experiment and should stay narrow.

This is separate from context creation. The point is to study actual GPU use across processes.

**Tasks**
- Run one interpretable two-process benchmark.
- Compare:
  - **without MPS**
  - **with MPS**
- Capture the same host-visible signals used elsewhere:
  - futex / poll / ioctl structure
  - visible context churn if present
  - interference / fairness effects
  - completion / wait behavior across processes

**Questions**
- What changes when concurrency is across **processes** rather than **threads**?
- Does MPS materially change the host-visible structure?
- Is there evidence of process-level serialization or fairness effects?
- What does Linux see under multi-process GPU sharing?

**Deliverable**
- Final **Multi-process GPU Use** subsection
- Short **no-MPS vs MPS** comparison
- 2–3 mechanism-level conclusions

---

### Priority 3 — Final scheduler comparison
The scheduler branch is already done except for one remaining comparison.

The only meaningful remaining scheduler task is:
- **4 threads pinned to 2 cores vs 4 threads pinned to 4 cores**

Everything else already exists and should be treated as completed experimental material.

**Tasks**
- Re-run the scheduler summary script if needed.
- Fold the 4-thread / 4-core result into the final scheduler interpretation.
- Write the final comparison against the 4-thread / 2-core case.

**Questions**
- How much calmer is **4 threads / 4 cores** than **4 threads / 2 cores**?
- What remained invariant?
- Does the dominant synchronization / completion signature persist?

**Deliverable**
- One final scheduler comparison note
- Integration into the finished **Scheduler / Wakeup Analysis** write-up

---

### Priority 4 — Minimal real serving-stack validation
This is the final Phase 2 grounding step.

Keep it small. The objective is correspondence, not a large serving study.

**Tasks**
- Bring up minimal vLLM or SGLang.
- Start with:
  - one small model
  - one client / one request
  - one clean trace
- Then do one small follow-up variation such as:
  - short vs longer context
  - or single vs light concurrency

**Questions**
- Which synthetic mechanism signatures reappear in the real stack?
- Which branch best explains the real host-visible behavior?
- What dominates in practice: submission, sync/wakeup, initialization, or memory behavior?

**Deliverable**
- Final **Real Workload Validation** subsection
- Small comparison between serving traces and synthetic signatures

---

### Priority 5 — Phase 2 synthesis
This is the handoff into Phase 3.

**Tasks**
- Pull together the final messages / analyses from earlier project chats for:
  - sync
  - submission
  - allocation lifetime
  - scheduler
- Add:
  - context lifetime
  - multi-process GPU use
  - serving validation
- Build one mechanism atlas:
  - phenomenon
  - host-visible signature
  - mechanism interpretation
  - OS visibility implication
  - Phase 3 implication
- Extract 3–5 defensible Phase 2 claims.
- Write one short Phase 3 bridge note.

**Deliverable**
- Final **Phase 2 synthesis note**
- **Phase 3 requirements / prototype direction** note

---

### Priority 6 — Supervisor meeting
This should be explicit.

**Tasks**
- Contact your professor and request a meeting before or around the Phase 3 start.
- Send a short summary ahead of time covering:
  - what is completed in Phase 2
  - what is still being closed
  - the likely Phase 3 direction
  - whether you are considering a short delay for bounded perf work

**Meeting goals**
- check that Phase 2 is sufficiently closed
- discuss whether the multi-process + serving validation is enough
- discuss whether a bounded perf detour is worthwhile
- discuss what the Phase 3 prototype should expose

**Deliverable**
- Meeting booked
- One-page discussion note prepared

---

## Optional perf extension

### Why it might be useful
A bounded perf pass could strengthen:
- driver-path interpretation
- internal serialization understanding
- host-side call-stack evidence

### Constraint
Perf only works on some lab nodes, and those nodes top out at a **GTX 1050 Ti**, while most Phase 2 results were obtained on **RTX 2080 / RTX 4080**-class machines.

### Does that difference matter?
Yes.

A 1050 Ti is not a direct proxy for a 2080 or 4080 because:
- driver behavior may differ by architecture / generation
- kernel launch / completion timing may differ
- concurrency and pressure signatures may differ in scale
- absolute waiting behavior may shift

### What it can still be used for
A 1050 Ti can still be useful for **qualitative supporting evidence**, for example:
- call-stack structure
- rough driver-path shape
- whether a serialization path exists
- supporting evidence for host-side mechanism interpretation

It should **not** replace the main Phase 2 platform or be used to overturn the main results.

### Recommendation
Discuss perf with your professor, but only treat it as worth delaying Phase 3 if:
1. context lifetime is closed
2. the multi-process GPU-use study is done
3. serving validation is done
4. the perf question is sharply bounded
5. your professor agrees it is worth the time

---

## Suggested 10-day breakdown

### Day 1–2
**Focus:** Context lifetime re-analysis  
**Output:** recovered tables, interpretation, write-up

### Day 3–4
**Focus:** Focused multi-process GPU-use study  
**Output:** no-MPS vs MPS comparison and conclusions

### Day 5
**Focus:** Final scheduler comparison  
**Output:** 4 threads / 2 cores vs 4 threads / 4 cores comparison integrated into scheduler write-up

### Day 6–7
**Focus:** Minimal serving-stack validation  
**Output:** one real-stack signature comparison

### Day 8
**Focus:** Pull completed branch analyses from earlier chats  
**Output:** consolidated Phase 2 notes base

### Day 9
**Focus:** Final synthesis  
**Output:** mechanism atlas + 3–5 claims + Phase 3 bridge

### Day 10
**Focus:** Supervisor meeting prep / booking + perf decision  
**Output:** one-page meeting note and explicit go/no-go decision on optional perf extension

---

## Final deliverables by the Phase 3 start

1. **Context lifetime** re-analysed and written up  
2. **One focused multi-process GPU-use study** completed  
3. **Final scheduler comparison** completed  
   - 4 threads / 2 cores vs 4 threads / 4 cores  
4. **Minimal real serving-stack validation** completed  
5. **Merged Phase 2 synthesis note** completed  
6. **3–5 defensible mechanism-level claims** written down  
7. **Phase 3 prototype direction note** completed  
8. **Supervisor meeting** booked or completed  
9. **Optional perf decision** made explicitly

---

## Definition of ready for Phase 3

You are ready for Phase 3 when you can state:
- what host-visible mechanisms dominate GPU interaction
- which effects come from submission, synchronization, memory lifetime, initialization, and multi-process sharing
- what Linux can observe
- what Linux cannot infer
- what semantic structure the Phase 3 abstraction should expose
