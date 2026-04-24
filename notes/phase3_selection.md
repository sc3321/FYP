# Phase 3 Narrowing Logbook

## 0. Project state
- Date:
- Current goal:
- Next deadline:
- Current main uncertainties:

---

## 1. Phase 2 pressure shortlist

### Pressure family 1
- Branch/source:
- What was observed:
- Why it matters:
- Why it felt strong:
- What host-visible mechanism seems involved:

### Pressure family 2
- Branch/source:
- What was observed:
- Why it matters:
- Why it felt strong:
- What host-visible mechanism seems involved:

### Pressure family 3
- Branch/source:
- What was observed:
- Why it matters:
- Why it felt strong:
- What host-visible mechanism seems involved:

---

## 2. Candidate missing semantics

### Candidate A
- Motivating pressure family:
- What ambiguity exists at the host boundary:
- What could the workload/runtime know that the OS does not:
- One extra truthful bit I might want to export:
- Rough semantic category:
  - grouping / phase / completion / criticality / other
- Why this might matter:
- Why this might be wrong:

### Candidate B
- Motivating pressure family:
- What ambiguity exists at the host boundary:
- What could the workload/runtime know that the OS does not:
- One extra truthful bit I might want to export:
- Rough semantic category:
  - grouping / phase / completion / criticality / other
- Why this might matter:
- Why this might be wrong:

### Candidate C
- Motivating pressure family:
- What ambiguity exists at the host boundary:
- What could the workload/runtime know that the OS does not:
- One extra truthful bit I might want to export:
- Rough semantic category:
  - grouping / phase / completion / criticality / other
- Why this might matter:
- Why this might be wrong:

---

## 3. vLLM validation plan

### Mechanism family 1
- Microbenchmark source:
- Actual mechanism isolated:
- Plausible vLLM analogue:
- What would count as supportive evidence:
- What would count as ambiguous evidence:
- What would count as weakening evidence:
- Scenarios to try:

### Mechanism family 2
- Microbenchmark source:
- Actual mechanism isolated:
- Plausible vLLM analogue:
- What would count as supportive evidence:
- What would count as ambiguous evidence:
- What would count as weakening evidence:
- Scenarios to try:

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
