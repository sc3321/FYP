# Experimental design 

## The 3 layer loop:

### Layer 1: Structural Dimensions (What I change)

Pertubation space for my experiments.

4 levers of change:

#### Submission Topology:

The how and when of work being handed to the GPU by the CPU.
Mechanically, this includes:
    1. Kernel launches calls (cudaLaunchKernel).
    2. Batched submission or separated submission.
    3. Fused kernels?
    4. How many launches.
    5. Multiple streams?

Answers how many times the host requests work to be enqueued to the device.

Each launch involves some runtime bookkeeping, driver calls, queue submission etc. This is all valuable OS level interpretation signals.

Possible perturbations:
    1. varying kernel sizes.
    2. 100 small kernels vs 1 large kernel.
    3. fused kernel implementation?

This lever helps isolate the overhead due to the dispatch of work.

#### Synchronisation Topology:

The how and when of CPU waiting for GPU.

Mechanically this includes:
    1. cudStreamSynchronise()
    2. cudaDeviceSynchronise()
    3. cudaMemcpy()
    4. Program exits.

This answers when does the host block waiting for the device to complete work. 

Waiting is implemented via futex(), epoll, poll, thread sleeps etc. which are all valuable OS visible mechanisms.

Possible perturbations:
    1. Syncing once at the end of the work.
    2. Syncing every iteration of work.
    3. No syncing.
    4. Asynchronous memcpy.

This lever helps isolate the dynamics of completion behaviour.

#### Allocation Lifetime:

The how and when of allocating and freeing device memory.

Mechanically it includes:
    1. cudaMalloc()
    2. cudaFree()
    3. Host pinned memory.
    4. unified memory.
    5. Reuse vs Per-Iteration allocation.

This answers when and how the device memory is managed throughout execution.
Memory management on device involved mmap, mprotect, pinned pages, ioctl calls etc. which are all valuable OS visible signals to explore. 

Possible Pertubations:
    1. Reuse buffers.
    2. Allocate once outside loop.
    3. Many small allocations and frees inside loop.
    4. Changing allocation size.

This lever isolates the overhead due to memory management.
 
#### Context Lifetime:

This is the creation and initialisation of the CUDA runtime.

Mechanically this means:
    1. First cuda call triggering context creation.
    2. Multiple processes creating contexts.
    3. Warmup?

When does the runtime setup the device state and memory addresses.
Involves large memory mapping, driver initialiation, thread creation etc. which are useful for OS level interpretation.

Possible Pertubations:
    1. Single vs multiple runs.
    2. Multiple processes?
    3. Warmup vs later launches?
    4. Early initialiation.

This lever isolates initialisation overheads.

### Layer 2- Syscalls (What I observe)

Syscalls are the primary measurement surface of the experiments. Gathering detailed statistics about how they vary across different structural changes made in Layer 1 is what will serve as the basis for conclusions and future hypotheses. 

### Layer 3- Interpretive layer.

Based on a given structural lever, and the observed syscall variation, this layer seeks to form a direct conclusion about a singular dimension of either:

    - Submission?
    - Completion?
    - Memory management?
    - Threading/Initialisation?


Here is a concrete example of an end-to-end experiment:

    Experiment: Synchronization Topology

    Version A:

    Sync once at end.

    Version B:

    Sync inside every iteration.

    You run strace -c.

    Observation:

    futex increases massively in Version B.

    ioctl roughly unchanged.

    mmap unchanged.

    Layer 3 reasoning:

    You ask:
    Which mechanism category is affected by synchronization topology?

    Answer:
    Completion behavior.

    So futex likely corresponds to:

    Host thread blocking until GPU work completes.

    Now you can state:

    Changing synchronization topology isolates futex as part of the GPU completion path.

    That’s a Phase 2 result.
