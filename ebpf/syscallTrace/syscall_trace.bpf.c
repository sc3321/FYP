#include "vmlinux.h"
#include "common.h"
#include <bpf/bpf_helpers.h>

char LICENSE[] SEC("license") = "GPL";

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 64);
    __type(key, unsigned int);
    __type(value, unsigned char);
} watched_tgids SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 64);
    __type(key, long);
    __type(value, unsigned int);
} watched_syscalls SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, unsigned int);
    __type(value, struct syscall_start);
} inflight SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24);
} events SEC(".maps");

SEC("tracepoint/raw_syscalls/sys_enter")
int handle_sys_enter(struct trace_event_raw_sys_enter *ctx)
{

    unsigned long long pid_tgid = bpf_get_current_pid_tgid();
    unsigned int tgid = pid_tgid >> 32;
    unsigned int tid = (unsigned int)pid_tgid;

    unsigned char *watch_pid = bpf_map_lookup_elem(&watched_tgids, &tgid);
    if (!watch_pid)
        return 0;

    long syscall_id = ctx->id;

    unsigned int *kind = bpf_map_lookup_elem(&watched_syscalls, &syscall_id);
    if (!kind)
        return 0;

    struct syscall_start st = {};
    st.start_ns = bpf_ktime_get_ns();
    st.syscall_id = syscall_id;
    st.kind = *kind;
    st.arg0 = ctx->args[0];
    st.arg1 = ctx->args[1];
    st.arg2 = ctx->args[2];

    bpf_map_update_elem(&inflight, &tid, &st, BPF_ANY);
    return 0;

}

SEC("tracepoint/raw_syscalls/sys_exit")
int handle_sys_exit(struct trace_event_raw_sys_exit *ctx)
{
    unsigned long long pid_tgid = bpf_get_current_pid_tgid();
    unsigned int tgid = pid_tgid >> 32;
    unsigned int tid = (unsigned int)pid_tgid;

    unsigned char *watch_pid = bpf_map_lookup_elem(&watched_tgids, &tgid);
    if (!watch_pid)
        return 0;

    struct syscall_start *st = bpf_map_lookup_elem(&inflight, &tid);
    if (!st)
        return 0;

    if (ctx->id != st->syscall_id) {
        bpf_map_delete_elem(&inflight, &tid);
        return 0;
    }

    unsigned long long end_ns = bpf_ktime_get_ns();

    struct syscall_event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        bpf_map_delete_elem(&inflight, &tid);
        return 0;
    }

    e->start_ns = st->start_ns;
    e->end_ns = end_ns;
    e->dur_ns = end_ns - st->start_ns;

    e->tgid = tgid;
    e->tid = tid;

    e->syscall_id = st->syscall_id;
    e->kind = st->kind;
    e->ret = ctx->ret;

    e->arg0 = st->arg0;
    e->arg1 = st->arg1;
    e->arg2 = st->arg2;

    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    bpf_ringbuf_submit(e, 0);
    bpf_map_delete_elem(&inflight, &tid);
    return 0;
}





