#ifndef SYSCALL_TRACE_COMMON_H
#define SYSCALL_TRACE_COMMON_H

#define KIND_UNKNOWN             0
#define KIND_IOCTL               1
#define KIND_FUTEX               2
#define KIND_NANOSLEEP           3
#define KIND_CLOCK_NANOSLEEP     4
#define KIND_POLL                5
#define KIND_PPOLL               6
#define KIND_EPOLL_WAIT          7
#define KIND_EPOLL_PWAIT         8
#define KIND_FUTEX_WAITV         9

struct syscall_start {
    unsigned long long  start_ns;
    long                syscall_id;
    unsigned int        kind;
    unsigned long long  arg0;
    unsigned long long  arg1;
    unsigned long long  arg2;
};

struct syscall_event {
    unsigned long long start_ns;
    unsigned long long end_ns;
    unsigned long long dur_ns;

    unsigned int tgid;
    unsigned int tid;

    long syscall_id;
    unsigned int kind;
    long ret;

    unsigned long long arg0;
    unsigned long long arg1;
    unsigned long long arg2;

    char comm[16];
};

#endif
