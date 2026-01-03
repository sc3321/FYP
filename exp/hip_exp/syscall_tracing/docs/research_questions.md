3 Research questions:

Q1(Repetition axis):
    Fixing an execution model(probably kernels) and memory footprint(bytes), how does the OS interaction scale with the iterations. 
    Measurment metrics will be:
        - calls/iteration.
        - syscall time/iteration.

Q2(Bytes axis):
    Fixing execution model(probably kernels) and iteration count(iterations), how does the OS interaction scale with the bytes value.
    Measurement metrics will be:
        - calls/byte allocated.
        - syscall time/ byte allocated.

Q3(Execution axis):
    Fixing the program parametres(iterations and bytes), how does the OS interaction scale/vary across the different execution models(baseline, kernel, alloc).
    Measurement metrics will be:
        - calls/iteration.
        - syscall time/iteration.


This will be plotted per syscall category and then explored in detail for specific syscalls for more granularity.
e.g synchronisation and then futex in more detail if it dominates the synchronisation category.

For every "frozen question", the main deliverables will be:

    1) A primary plot/table which includes information at the syscall category level.
    2) Perhaps a secondary/tertiary plot or table drilling into specific syscalls.
    3) Writings in report style of the findings themselves- focussing on answering the question on purely observation of the data.
    4) A final "Claim"*

*claim: This claim is an attempt at a hypothesis at the generalisation of OS interaction with GPU programs given the observed behaviour. A good claim states some dependency, some insensitivity(elimination), some control behaviour and eludes to a phase 2 actionable task. 

Perhaps rather than a final claim, it may be better to get the 3 deliverables and then come up with 3-6 claims using all of the gathered information.










