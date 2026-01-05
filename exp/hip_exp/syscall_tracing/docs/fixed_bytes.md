# Byes exploration

Upon fixing the execution models to kernels and the iteration count to 100000, the purpose of this experiment was to gather the variation of OS interaction with scaling the memory footprint of the program.

The 2 primary metrics for evaluating this interaction between the program and the OS was: syscall count / kilobyte and usec_per_category/ kilobyte. 

## Results

The tables below, showcase the results that were gathered by varying the bytes from 1000000 -> 10000000:

syscall count/ kilobyte:

    bytes	    control_io	file_io	memory	other	sync
    1000000	    0.002	    0.096	0.388	0.06	0.02
    2500000	    0.0004	    0.0192	0.0776	0.012	0.0044
    5000000	    0.0004	    0.0192	0.0776	0.012	0.0038
    7500000	    0.00013	    0.0064	0.02573	0.004	0.0012
    10000000	0.0002	    0.0096	0.0388	0.006	0.0021

usec_per_sycall/ kilobyte:
    
    bytes	    control_io	file_io	memory	other	sync
    1000000	    0.011	    3.147	24.82	3.448	1679.672
    2500000	    0.0004	    0.5684	3.4028	0.3192	183.36
    5000000	    0.028	    0.3906	3.9546	0.3284	388.8558
    7500000	    0.00253	    0.18133	1.18146	0.17946	153.21426
    10000000	0.0001	    0.3129	3.2745	0.1278	251.7502

## Analysis

Firstly, consideration of the syscalls / kilobye metric indicates a strong amortization of any memory footprint related syscall interaction. It seems that there is some initial, possible fixed cost of OS interaction which rapidly tends towards 0 as the byte size increases. This is consistent with the syscalls being involved in structural tasks like setup and teardown of the program. 

Concluding that the number of absolute syscalls does not scale with increasing byte involvement, the usec / byte metric can be used to analyse the actual performance effect of each syscall to the program execution.

Almost all of the categories tend towards 0, indicating that they do not likely occur on the program steady state execution path. They are involved only a fixed number of times, and have almost no relationship to the bytes that are involved with the program. Comparing their time in execution to the bytes only causes a decreasing relationship that approaches 0.:

    T(bytes) = C / B where C would represent some fixed cost for almost every run. Diving by an increased byte number only causes T(bytes) -> 0.

sync remains still the exception to this rule by settling towards a non-negligible value and remains erratic even at higher byte values. This indicates that sync is also NOT sensitive to the byte values as it would show some stabilisation or consistent increase...

This lack of observable relationship forces the conclusion that the sync category is also not byte sensitive, but does remain on the program steady state path. Without consideration of other categories, we can conclude that it is event or phase driven resulting in persistent OS involvement regardless of byte size.

It may seem counter-intuitive that the increasing memory footprint has no effect on the syscalls, and if the OVERALL program execution time was measured, it would show an immediate and obvious increase- after all we are simply operating on more data hence more operations. The key distinction however, is that we are measuring the time the OS is involved with the program by probing syscall time. The time spent in syscalls related to synchronisation is actually independent to the GPU work itself. The syscalls appear structurally in the program flow, and only for coordination, i.e when blocking or waking up a CPU thread which is waiting for GPU work to complete. The OS is called into action only for the blocking or waking procedure which is independent to the size of data. The number of absolute calls remains constant between 10-20 futex calls independent of memory footprint, however the time of each call does increase slightly with memory footprint but not nearly enough to indicate byte sensitivity. 

#### Summary of Analysis:

            Execution is divided into a small number of phases

            OS interaction occurs only at phase boundaries

            The number of phase boundaries is independent of bytes

            Increasing bytes increases work inside a phase(GPU execution), not the number of phases

            Synchronisation syscalls are few but long because they block across a phase

            Their frequency is structural; their duration reflects coarse waiting* plus OS overhead

            Neither frequency nor duration scales proportionally with bytes

    *coarse waiting: Very roughly proportional to the length of time spent in the phase. OS overhead still dominates.


## Claim:

    For a given execution model and iteration count, the OS involvement with program execution is mostly insensitive to the memory footprint of the program. The syscalls / kilobyte metric clearly indicates effective amortization of most syscall categories, showing they are all insensitive to the memory footprint but rather due to general OS overheads. The usec_per_sycall / kilobyte shows a very similar story, with the exception again of the sync related syscalls. These syscalls dominate the OS time across all memory footprints but do not still indicate a sensitivity to byte values due to their unstable values. Further probing shows that the number of syscalls remains mostly constant at around 10-20 syscalls and do not scale with increasing memory footprint. There is perhaps some very coarse timing sensitivity with regards to work completion (particularly for GPU operations) which seems to increase with memory footprint but it is not conclusive.

