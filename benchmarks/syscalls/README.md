Unix systems implement most interfaces between User Mode processes and hardware
devices by means of system calls issued to the kernel.

Applications run in USER-MODE and when the system call is handled by the kernel the system will enter KERNEL MODE.

This distinction is critical and is the basis of the first benchmark that I think is important to run. 

System calls represent a class of instructions which do the following:
    - User-kernel mode transition.
    - The kernel implementation (handling) of whatever the systemcall does
    - kernel mode-user mode return.

This transition between user mode and kernel mode is not trivial and involves a lot of work like saving user registers, switching this privilage level, storing stack variables, doing the kernel work of the systemcall and then switching back to user level. This fundamental round-trip time makes sense to measure first.

To make this a fair experiment, it is perhaps best to make the actual kernel work as minimal as possible, so by benchmarking something like getpid() which is a syscall which just returns a process id (negligible work) the time profiled is isolated to all the overhead of switching to and from a syscall.



