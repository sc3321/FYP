# Experimental design 

## The 3 layer loop:

### Layer 1: Structural Dimensions (What I change)

Pertubation space for my experiments.

4 levers of change:

#### Submission Topology:

The how and when of work being handed to the GPU by the CPU.

Mechanically, this includes:<br>
    1. Kernel launches calls (cudaLaunchKernel).<br>
    2. Batched submission or separated submission.<br>
    3. Fused kernels?<br>
    4. How many launches.<br>
    5. Multiple streams?<br>

Answers how many times the host requests work to be enqueued to the device.

Each launch involves some runtime bookkeeping, driver calls, queue submission etc. This is all valuable OS level interpretation signals.

Possible perturbations:<br>
    1. varying kernel sizes.<br>
    2. 100 small kernels vs 1 large kernel.<br>
    3. fused kernel implementation?<br>

This lever helps isolate the overhead due to the dispatch of work.

#### Synchronisation Topology:

The how and when of CPU waiting for GPU.

Mechanically this includes:<br>
    1. cudStreamSynchronise()<br>
    2. cudaDeviceSynchronise()<br>
    3. cudaMemcpy()<br>
    4. Program exits.<br>

This answers when does the host block waiting for the device to complete work. 

Waiting is implemented via futex(), epoll, poll, thread sleeps etc. which are all valuable OS visible mechanisms.

Possible perturbations:<br>
    1. Syncing once at the end of the work.<br>
    2. Syncing every iteration of work.<br>
    3. No syncing.<br>
    4. Asynchronous memcpy.<br>

This lever helps isolate the dynamics of completion behaviour.

#### Allocation Lifetime:

The how and when of allocating and freeing device memory.

Mechanically it includes:<br>
    1. cudaMalloc()<br>
    2. cudaFree()<br>
    3. Host pinned memory.<br>
    4. unified memory.<br>
    5. Reuse vs Per-Iteration allocation.<br>

This answers when and how the device memory is managed throughout execution.
Memory management on device involved mmap, mprotect, pinned pages, ioctl calls etc. which are all valuable OS visible signals to explore. 

Possible Pertubations:<br>
    1. Reuse buffers.<br>
    2. Allocate once outside loop.<br>
    3. Many small allocations and frees inside loop.<br>
    4. Changing allocation size.<br>

This lever isolates the overhead due to memory management.
 
#### Context Lifetime:

This is the creation and initialisation of the CUDA runtime.

Mechanically this means:<br>
    1. First cuda call triggering context creation.<br>
    2. Multiple processes creating contexts.<br>
    3. Warmup?<br>

When does the runtime setup the device state and memory addresses.
Involves large memory mapping, driver initialiation, thread creation etc. which are useful for OS level interpretation.

Possible Pertubations:<br>
    1. Single vs multiple runs.<br>
    2. Multiple processes?<br>
    3. Warmup vs later launches?<br>
    4. Early initialiation.<br>

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
