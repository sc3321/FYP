# Variant exploration:

Fixing the iteration count to 100000 and the memory footprint to 10MB, the purpose of this experiment was to measure the variant of OS involvement with the 3 different program execution models: alloc, kernels and baseline.

It is worth mentioning that these results were gathered against HIP_CPU which is not a real HIP runtime that interfaces with a GPU device. The hope was that while HIP-CPU does not involve real runtime drivers, it would still preserve the execution policies of a real GPU making this a useful investigation.

The primary metrics that I used to investigate this relationship was usec/ iteration and calls/ iteration. 

## Results:

calls/ iteration:

    variant	    control_io	file_io	memory	other	sync
    
    alloc	    1.00E-05	0.00017	0.00045	0.00018	1.00026
    baseline	1.00E-05	0.00017	0.00046	0.00018	4.00E-05
    kernels	    2.00E-05	0.00096	0.00388	0.0006	0.00021

usec/ iteration:

    variant	    control_io	file_io	memory	other	sync
    
    alloc	    0.00161	    0.00036	0.17591	0.00429	239.73847
    baseline	0.00099	    0.006	0.11407	0.01218	0.00382
    kernels	    1.00E-05	0.03129	0.32745	0.01278	25.17502

## Analysis:

Calls per iteration is an insiteful metric into how the OS interaction scales with the program execution flow. It is quite revealing as to which syscall categories are actually involved in the program steady state path and which remain as setup overheads. It is clear from the results that the majority of categories are in the latter group, with calls/ iteration being incredibly low across all the execution variants. The exception here is the syncronisation syscalls in the alloc variant of the program which stabilises very nicely to 1. This is quite a profound figure, and indicates that a synchronisation syscall occurs at almost every single iteration of work. The OS is most certainly involved in the steady state execution of this program profile.

Analysis of the usec/ iteration results shows us that for bytes=10MB and 100k iterations, baseline incurs ~0.137 µs/iter total syscall time, whereas alloc incurs ~240 µs/iter and kernels ~25.5 µs/iter, corresponding to ~1750× and ~186× increases over baseline respectively. This immediately tells us that the GPU runtime introduces OS overhead. Further probing shows us that in both runtime variants, synchronization dominates steady-state syscall time (alloc: 99.92%, kernels: 98.55%), but with opposite regimes: alloc issues ~1 sync syscall/iter (~240 µs per call) while kernels issues ~2.1e-4 sync syscalls/iter (~120 ms per call).

This tells us that for kernel workloads, syncronisation happens rarely but are heavyweight operations that occur at sync points. Allocation defined workloads always require a syncronisation but are comparitavely faster. The real distinction is that in the kernel workloads the OS is needed to arbiter the program progress at a few select syncronisation points, but this is not the case for allocation workloads where they are involved at every iteration.

## Claim

Keeping iteration count and memory footprint constant, distinct patterns of OS interaction emerge across different execution models. The CPU baseline exhibits minimal OS involvement, with syscalls well amortized and absent from the steady-state execution path.

In contrast, runtime-mediated execution models exhibit substantial OS involvement, but with structurally different regimes. Allocation-heavy workloads impose fine-grained OS interaction, with synchronization occurring approximately once per iteration, resulting in frequent but comparatively inexpensive syscalls that dominate total runtime. Kernel-based workloads, which are representative of AI execution patterns, instead amortize OS interaction, invoking synchronization rarely but at high cost, resulting in coarse-grained OS involvement that dominates latency rather than frequency.

These results demonstrate that OS interaction is governed primarily by execution structure and synchronization granularity rather than data volume, revealing two distinct regimes of fine- and coarse-grained OS involvement.
