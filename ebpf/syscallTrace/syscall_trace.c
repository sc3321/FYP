#include <bpf/libbpf.h>
#include <bpf/bpf.h>

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "common.h"
#include "syscall_trace.skel.h"

static volatile sig_atomic_t exiting = 0;

static void handle_signal(int signo)
{
    exiting = 1;
}

static const char *kind_name(unsigned int kind)
{
    switch (kind) {
    case KIND_IOCTL:           return "ioctl";
    case KIND_FUTEX:           return "futex";
    case KIND_NANOSLEEP:       return "nanosleep";
    case KIND_CLOCK_NANOSLEEP: return "clock_nanosleep";
    case KIND_POLL:            return "poll";
    case KIND_PPOLL:           return "ppoll";
    case KIND_EPOLL_WAIT:      return "epoll_wait";
    case KIND_EPOLL_PWAIT:     return "epoll_pwait";
    case KIND_FUTEX_WAITV:     return "futex_waitv";
    default:                   return "unknown";
    }
}

static int add_syscall(struct syscall_trace_bpf *skel, long nr,
                       unsigned int kind, const char *name)
{
    int err;

    if (nr < 0) {
        return 0;
    }

    err = bpf_map_update_elem(
        bpf_map__fd(skel->maps.watched_syscalls),
        &nr,
        &kind,
        BPF_ANY
    );

    if (err) {
        fprintf(stderr, "failed to watch syscall %s (%ld): %s\n",
                name, nr, strerror(errno));
        return err;
    }

    fprintf(stderr, "watching syscall %-22s nr=%ld\n", name, nr);
    return 0;
}

static int add_default_syscalls(struct syscall_trace_bpf *skel)
{
    int err = 0;

#ifdef __NR_ioctl
    err |= add_syscall(skel, __NR_ioctl, KIND_IOCTL, "ioctl");
#endif

#ifdef __NR_futex
    err |= add_syscall(skel, __NR_futex, KIND_FUTEX, "futex");
#endif

#ifdef __NR_futex_time64
    err |= add_syscall(skel, __NR_futex_time64, KIND_FUTEX, "futex_time64");
#endif

#ifdef __NR_futex_waitv
    err |= add_syscall(skel, __NR_futex_waitv, KIND_FUTEX_WAITV, "futex_waitv");
#endif

#ifdef __NR_nanosleep
    err |= add_syscall(skel, __NR_nanosleep, KIND_NANOSLEEP, "nanosleep");
#endif

#ifdef __NR_clock_nanosleep
    err |= add_syscall(skel, __NR_clock_nanosleep, KIND_CLOCK_NANOSLEEP, "clock_nanosleep");
#endif

#ifdef __NR_clock_nanosleep_time64
    err |= add_syscall(skel, __NR_clock_nanosleep_time64, KIND_CLOCK_NANOSLEEP, "clock_nanosleep_time64");
#endif

#ifdef __NR_poll
    err |= add_syscall(skel, __NR_poll, KIND_POLL, "poll");
#endif

#ifdef __NR_ppoll
    err |= add_syscall(skel, __NR_ppoll, KIND_PPOLL, "ppoll");
#endif

#ifdef __NR_epoll_wait
    err |= add_syscall(skel, __NR_epoll_wait, KIND_EPOLL_WAIT, "epoll_wait");
#endif

#ifdef __NR_epoll_pwait
    err |= add_syscall(skel, __NR_epoll_pwait, KIND_EPOLL_PWAIT, "epoll_pwait");
#endif

    return err;
}

static int handle_event(void *ctx, void *data, size_t data_sz)
{
    const struct syscall_event *e = data;

    printf("{\"start_ns\":%llu,"
           "\"end_ns\":%llu,"
           "\"dur_ns\":%llu,"
           "\"dur_us\":%.3f,"
           "\"tgid\":%u,"
           "\"tid\":%u,"
           "\"syscall_id\":%ld,"
           "\"kind\":\"%s\","
           "\"ret\":%ld,"
           "\"arg0\":%llu,"
           "\"arg1\":%llu,"
           "\"arg2\":%llu,"
           "\"comm\":\"%s\"}\n",
           e->start_ns,
           e->end_ns,
           e->dur_ns,
           (double)e->dur_ns / 1000.0,
           e->tgid,
           e->tid,
           e->syscall_id,
           kind_name(e->kind),
           e->ret,
           e->arg0,
           e->arg1,
           e->arg2,
           e->comm);

    return 0;
}

int main(int argc, char **argv)
{
    struct syscall_trace_bpf *skel = NULL;
    struct ring_buffer *rb = NULL;
    int err = 0;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <pid> [pid...]\n", argv[0]);
        fprintf(stderr, "example: sudo %s 12345 12346 > events.jsonl\n", argv[0]);
        return 1;
    }

    setvbuf(stdout, NULL, _IOLBF, 0);

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    skel = syscall_trace_bpf__open_and_load();
    if (!skel) {
        fprintf(stderr, "failed to open/load BPF skeleton\n");
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        unsigned int tgid = (unsigned int)strtoul(argv[i], NULL, 10);
        unsigned char enabled = 1;

        err = bpf_map_update_elem(
            bpf_map__fd(skel->maps.watched_tgids),
            &tgid,
            &enabled,
            BPF_ANY
        );

        if (err) {
            fprintf(stderr, "failed to watch pid %u: %s\n",
                    tgid, strerror(errno));
            goto cleanup;
        }

        fprintf(stderr, "watching pid %u\n", tgid);
    }

    err = add_default_syscalls(skel);
    if (err) {
        goto cleanup;
    }

    err = syscall_trace_bpf__attach(skel);
    if (err) {
        fprintf(stderr, "failed to attach BPF programs: %d\n", err);
        goto cleanup;
    }

    rb = ring_buffer__new(
        bpf_map__fd(skel->maps.events),
        handle_event,
        NULL,
        NULL
    );

    if (!rb) {
        fprintf(stderr, "failed to create ring buffer\n");
        err = 1;
        goto cleanup;
    }

    fprintf(stderr, "tracing selected syscall durations. Ctrl-C to stop.\n");

    while (!exiting) {
        err = ring_buffer__poll(rb, 100);
        if (err < 0 && err != -EINTR) {
            fprintf(stderr, "ring_buffer__poll error: %d\n", err);
            break;
        }
        err = 0;
    }

cleanup:
    ring_buffer__free(rb);
    syscall_trace_bpf__destroy(skel);
    return err != 0;
}
