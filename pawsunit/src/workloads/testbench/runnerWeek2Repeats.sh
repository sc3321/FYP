#!/usr/bin/env bash
set -euo pipefail

BIN="./out/gpu_phase_worker"
OUT="./out/output_week2_repeats"

TRACE_SYSCALLS="futex,poll,ppoll,epoll_wait,ioctl,nanosleep,clock_nanosleep,mmap,munmap,mprotect,write"

LC_ITERS=300
BE_ITERS=50
CHUNKS=16
LC_SLEEP_US=1000

REPEATS=3

rm -rf "$OUT"
mkdir -p "$OUT"

write_config() {
    local dir="$1"

    cat > "$dir/config.txt" <<EOF
BIN=$BIN
TRACE_SYSCALLS=$TRACE_SYSCALLS
LC_ITERS=$LC_ITERS
BE_ITERS=$BE_ITERS
CHUNKS=$CHUNKS
LC_SLEEP_US=$LC_SLEEP_US
REPEATS=$REPEATS
EOF
}

run_single() {
    local name="$1"
    shift

    local dir="$OUT/$name"
    local event_dir="$dir/events"
    local strace_dir="$dir/strace"

    rm -rf "$dir"
    mkdir -p "$event_dir" "$strace_dir"

    write_config "$dir"

    echo
    echo "[RUN] $name"

    GPU_PHASE_LOG_DIR="$event_dir" \
    strace -ff -ttt -T \
        -e trace="$TRACE_SYSCALLS" \
        -s 128 \
        -o "$strace_dir/trace" \
        "$BIN" "$@" \
        > "$dir/stdout.txt" \
        2> "$dir/stderr.txt"
}

run_lc_vs_be() {
    local name="$1"
    local be_mode="$2"
    local n_be="$3"

    local dir="$OUT/$name"
    local lc_event_dir="$dir/lc_events"
    local strace_dir="$dir/strace"

    rm -rf "$dir"
    mkdir -p "$lc_event_dir" "$strace_dir"

    write_config "$dir"

    echo
    echo "[RUN] $name"

    local be_pids=()

    # Start BE first so that LC is more likely to overlap active BE work.
    for i in $(seq 1 "$n_be"); do
        local be_event_dir="$dir/be${i}_events"
        mkdir -p "$be_event_dir"

        GPU_PHASE_LOG_DIR="$be_event_dir" \
        strace -ff -ttt -T \
            -e trace="$TRACE_SYSCALLS" \
            -s 128 \
            -o "$strace_dir/be${i}.trace" \
            "$BIN" \
                --class BE \
                --mode "$be_mode" \
                --iters "$BE_ITERS" \
                --chunks "$CHUNKS" \
            > "$dir/be${i}.stdout.txt" \
            2> "$dir/be${i}.stderr.txt" &

        be_pids+=("$!")
    done

    # Small delay lets BE enter its loop before LC begins.
    sleep 0.2

    GPU_PHASE_LOG_DIR="$lc_event_dir" \
    strace -ff -ttt -T \
        -e trace="$TRACE_SYSCALLS" \
        -s 128 \
        -o "$strace_dir/lc.trace" \
        "$BIN" \
            --class LC \
            --mode lc \
            --iters "$LC_ITERS" \
            --sleep-us "$LC_SLEEP_US" \
        > "$dir/lc.stdout.txt" \
        2> "$dir/lc.stderr.txt" &

    local lc_pid=$!

    local status=0

    if ! wait "$lc_pid"; then
        echo "[ERROR] LC process failed in $name" >&2
        status=1
    fi

    for pid in "${be_pids[@]}"; do
        if ! wait "$pid"; then
            echo "[ERROR] BE process $pid failed in $name" >&2
            status=1
        fi
    done

    return "$status"
}

for r in $(seq 1 "$REPEATS"); do
    run_single "lc_alone_stable_r${r}" \
        --class LC \
        --mode lc \
        --iters "$LC_ITERS" \
        --sleep-us "$LC_SLEEP_US"

    run_single "be_long_alone_r${r}" \
        --class BE \
        --mode be-long \
        --iters "$BE_ITERS" \
        --chunks "$CHUNKS"

    run_single "be_chunked_alone_r${r}" \
        --class BE \
        --mode be-chunked \
        --iters "$BE_ITERS" \
        --chunks "$CHUNKS"

    run_lc_vs_be "lc_vs_4_be_long_r${r}" "be-long" 4
    run_lc_vs_be "lc_vs_1_be_chunked_r${r}" "be-chunked" 1
done

echo
echo "Done. Repeated Week 2 results written to:"
echo "$OUT"
