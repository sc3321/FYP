# Iteration exploration

Upon fixing the execution models to kernels and the memory footprint to 10MB, the purpose of this experiment was to gather the variation of OS interaction with scaling the iteration counts.

The 2 primary metrics for evaluating this interaction between the program and the OS was: syscall count / iteration and usec_per_sycall / iteration. 

## Results

The tables below, showcase the results that were gathered by varying the iterations from 100 -> 100,000:

syscall count/ iteration:

    iterations	control_io	file_io	memory	other	sync

    100	        0.01        0.48	0.67	0.3	    0.1
    5000	    0.0002	    0.0096	0.0146	0.006	0.002
    10000	    0.0001	    0.0048	0.0081	0.003   0.0011
    50000	    2.00E-05	0.00096	0.00266	0.0006	0.00022
    100000	    1.00E-05	0.00048	0.00194	0.0003	0.0001

usec_per_sycall/ iteration:

    iterations	control_io	file_io	memory	other	sync

    100	        1.46	    17.24	40.56	8.2	    60.61
    5000	    0.0014	    0.1462	0.6454	0.1036	13.2882
    10000	    0.0015	    0.057	0.5804	0.0572  12.5679
    50000	    0	        0.02704	0.18284	0.01282 13.20224
    100000	    0	        0.01683	0.20494	0.00642 14.07721

## Analysis

Disappointingly, considering the syscall count/iteration does not deliver much novel information to us. It shows consistently, that the syscall count amortizes over multiple iterations of work. Almost all of the 5 considered categories of syscall tend towards 0 as the iterations increases. 

Amortization in the context of OS and syscalls is when: 
    Preliminary fixed costs are paid for kernel transition/ setup. This fixed cost is paid once and then reused across many iterations. 

Considering the usec_per_syscall/ iteration metric yields rather more interesting results. It is seen that for most of the syscall categories, there is a tend towards 0 as the iteration count increases. This is indicative of good amortization across the categories. It shows that regardless of the absolute call amount, the actual performance cost of each OS interaction, when factored across the entire program execution path, tends towards 0. This is only untrue for the sync category which shows an immediate collapse from ~60us to around ~13us. As the iterations increase however, this 13us remains mostly constant.

This constant trend over magnitudes of increase in the iteration count, indicates importantly that the sync related syscalls are involved in the steady state path of the program execution. 

    Steady state path: This is achieved when all the required kernel objects are created, address spaces mapped, control transitions are all mapped and warm. 

For things like the memory or control, they pertain to object and structure creation which happen only very rarely in the steady state(or not at all) and are mainly involved during program startup and completion. The fact that sync remains constant, indicates that the OS sync syscalls are involved in the program loop for things like coordination and observing progress. 

If investigated further, it can be seen that the absolute number of sync related calls actually remains constant: ~10. The fact that the time/iteration also remains constant as iterations increases so significantly indicates that each call is in fact much more expensive.

## Final Claim

    When fixing the execution model and the memory footprint, is is observed that the effect of most syscall categories exhibit textbook amortization. As the iteration counts increase, calls / iteration and usec / iteration for most categories tend towards 0, indicating that these syscalls are involved in setup and bookeeping activity which is spread over execution paths of the program. The syncronisation category of syscall remains the exception. This category, primarily dominated by the futex syscall, remains constant at ~10 calls made per program run. It is also observed that the usec / iteration stabilises quickly to a non-negligible value for all increasing iterations. The combination of these observations implies that the total amount of time spent in syncronisation related OS interaction increases linearly with the iteration count. Given the constant number of calls, this also indicates that the time spent per call also increases linearly with iteration count. We can conclude therefore that the sync events become progressively more expensive, and remain on the steady state critical path of the program scaling with the amount of work (iterations) performed between synchronisation events. 








