\chapter{Roadmap}

This section outlines progress to date and the direction of the project across subsequent phases. The project is structured to first establish a rigorous empirical understanding of OS-visible GPU behaviour, before progressing towards mechanism design and prototyping informed by those observations. 

\subsection{Progress to Date}

The following progress has been made at the time of writing:

\begin{itemize}
    \item A detailed literature review has been conducted covering GPU execution models, operating-system visibility boundaries, and prior systems for GPU scheduling and sharing. This research is documented in the background section of this report. 
    \item A configurable C++ microbenchmark harness has been implemented to exercise GPU-backed computation under controlled conditions.
    \item Multiple execution backends have been integrated, including CPU, CUDA, and HIP\_CPU. The harness also supports AMD GPU's HIP backend but it is yet to be tested with the goal of identifying vendor specific behaviour. 
    \item OS-level tracing infrastructure has been developed to capture system calls, memory-mapping activity, synchronisation events, and host-side timing behaviour.
    \item Preliminary Phase~1 experiments characterise OS involvement under varying workload parametres.
\end{itemize}

These components establish the experimental foundation required for the remaining phases of the project.

\subsection{Phase 2: Mechanism-Isolation Experiments}

Phase~2 focuses on isolating and stress-testing the mechanisms identified during Phase~1. Building on the initial characterisation of OS-visible behaviour, this phase introduces targeted interventions to better understand causal relationships between workload parameters and OS-level signals.

Planned activities include controlled variation of submission cadence, synchronisation frequency, and execution ordering to determine which observed behaviours are invariant and which are sensitive to specific execution patterns. These experiments aim to distinguish incidental correlations from meaningful execution dependencies, and to identify the limits of inference that can be drawn from OS-visible signals alone.

The outcome of Phase~2 will be a refined understanding of which execution dimensions are both observable and stable enough to support higher-level abstractions.

\subsection{Phase 3: User-Space Abstraction Prototype}

Phase~3 explores the feasibility of introducing lightweight abstractions at the OS boundary to recover semantic information lost in current GPU execution stacks. Rather than proposing invasive kernel or driver modifications, this phase focuses on a minimal, OS-adjacent prototype implemented primarily in C/C++.

The prototype will aim to expose structured execution information—such as phase boundaries or workload class hints—to the OS, enabling improved correlation between host-side signals and device execution behaviour. Potential implementations include a user-space daemon, structured runtime hints, or limited kernel-facing interfaces, depending on insights gained during Phase~2.

This phase is intended to evaluate whether modest interface extensions can meaningfully improve OS-level reasoning about GPU workloads without compromising portability or introducing significant overhead.

\subsection{Fallbacks and Extensions}

Given the exploratory nature of this work, several fallback paths are identified. If OS-visible signals prove too noisy or unstable to support meaningful abstraction, the project will focus instead on formally characterising these limitations and identifying where inference fundamentally breaks down. Alternatively, additional workload axes or tracing mechanisms may be introduced to extend the Phase~1 methodology.

Where time permits, extensions may include limited kernel-level instrumentation or validation across additional real-world workloads to strengthen generality.
