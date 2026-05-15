#!/usr/bin/env bash
set -euo pipefail

BIN="./out/equalWorkload"
OUT="./out/output_week2_warm_clean_equal_workload/"

TRACE_SYSCALLS="futex,poll,ppoll,epoll_wait,ioctl,nanosleep,clock_nanosleep,mmap,munmap,mprotect,write"

LC_ITERS=300
BE_ITERS=50
CHUNKS=16
LC_SLEEP_US=1000

# Warmup controls. These runs are NOT traced and are NOT analysed.
# They exist only to reduce first-use/context/clock effects before each measured run.
WARMUP_LC_ITERS=40
WARMUP_BE_ITERS=8
WARMUP_ROUNDS=2
COOLDOWN_SLEEP=2

rm -rf "$OUT"
mkdir -p "$OUT"

write_config() {
    local dir="$1"
    cat > "$dir/config.txt" <<EOF2
BIN=$BIN
OUT=$OUT
TRACE_SYSCALLS=$TRACE_SYSCALLS
LC_ITERS=$LC_ITERS
BE_ITERS=$BE_ITERS
CHUNKS=$CHUNKS
LC_SLEEP_US=$LC_SLEEP_US
WARMUP_LC_ITERS=$WARMUP_LC_ITERS
WARMUP_BE_ITERS=$WARMUP_BE_ITERS
WARMUP_ROUNDS=$WARMUP_ROUNDS
COOLDOWN_SLEEP=$COOLDOWN_SLEEP
EOF2
}

warm_gpu() {
    # Run a small mix of LC, long BE, and chunked BE without strace and without event logs.
    # Use a temporary event directory outside the analysed output tree, then delete it.
    local tmpdir
    tmpdir="$(mktemp -d /tmp/gpu_phase_warmup.XXXXXX)"

    for _ in $(seq 1 "$WARMUP_ROUNDS"); do
        GPU_PHASE_LOG_DIR="$tmpdir/lc" \
            "$BIN" --class LC --mode lc --iters "$WARMUP_LC_ITERS" --sleep-us "$LC_SLEEP_US" \
            > /dev/null 2>&1 || true

        GPU_PHASE_LOG_DIR="$tmpdir/be_long" \
            "$BIN" --class BE --mode be-long --iters "$WARMUP_BE_ITERS" --chunks "$CHUNKS" \
            > /dev/null 2>&1 || true

        GPU_PHASE_LOG_DIR="$tmpdir/be_chunked" \
            "$BIN" --class BE --mode be-chunked --iters "$WARMUP_BE_ITERS" --chunks "$CHUNKS" \
            > /dev/null 2>&1 || true
    done

    rm -rf "$tmpdir"
    sleep "$COOLDOWN_SLEEP"
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
    echo "======================================================================"
    echo "[WARMUP] before $name"
    echo "======================================================================"
    warm_gpu

    echo
    echo "======================================================================"
    echo "[RUN] $name"
    echo "======================================================================"

    GPU_PHASE_LOG_DIR="$event_dir" \
    strace -ff -ttt -T \
        -e trace="$TRACE_SYSCALLS" \
        -s 128 \
        -o "$strace_dir/trace" \
        "$BIN" "$@" \
        > "$dir/stdout.txt" \
        2> "$dir/stderr.txt"

    sleep "$COOLDOWN_SLEEP"
}

run_lc_alone() {
    local name="$1"
    run_single "$name" \
        --class LC \
        --mode lc \
        --iters "$LC_ITERS" \
        --sleep-us "$LC_SLEEP_US"
}

run_be_alone() {
    local name="$1"
    local be_mode="$2"
    run_single "$name" \
        --class BE \
        --mode "$be_mode" \
        --iters "$BE_ITERS" \
        --chunks "$CHUNKS"
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
    echo "======================================================================"
    echo "[WARMUP] before $name"
    echo "======================================================================"
    warm_gpu

    echo
    echo "======================================================================"
    echo "[RUN] $name"
    echo "BE mode: $be_mode"
    echo "BE workers: $n_be"
    echo "======================================================================"

    local be_pids=()

    # Start BE first so LC enters an already-active BE background.
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

    # Allow BE workers to enter their steady execution loop before LC starts.
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

    sleep "$COOLDOWN_SLEEP"
    return "$status"
}

log_note() {
    echo "$*" | tee -a "$OUT/run_notes.txt"
}

log_note "Warm-clean Week 2 run started: $(date)"
log_note "Output: $OUT"
log_note "Binary: $BIN"
log_note "This script deletes and recreates $OUT. Old datasets are untouched unless they use this same path."

################################################################################
# 1. Warmed LC-alone stability check
################################################################################

for r in 1 2 3 4 5; do
    run_lc_alone "lc_alone_warm_stability_r${r}"
done

################################################################################
# 2. Warmed BE-alone references
################################################################################

for r in 1 2 3; do
    run_be_alone "be_long_alone_warm_r${r}" "be-long"
    run_be_alone "be_chunked_alone_warm_r${r}" "be-chunked"
done

################################################################################
# 3. Bracketed paired LC-alone vs LC + 4 long BE
#    before/after LC-alone checks help detect drift/run-order contamination.
################################################################################

for p in 1 2 3; do
    run_lc_alone "pair${p}_long4_lc_alone_before"
    run_lc_vs_be "pair${p}_long4_lc_vs_4_be_long" "be-long" 4
    run_lc_alone "pair${p}_long4_lc_alone_after"
done

################################################################################
# 4. Bracketed paired LC-alone vs LC + 1 chunked BE
################################################################################

for p in 1 2 3; do
    run_lc_alone "pair${p}_chunk1_lc_alone_before"
    run_lc_vs_be "pair${p}_chunk1_lc_vs_1_be_chunked" "be-chunked" 1
    run_lc_alone "pair${p}_chunk1_lc_alone_after"
done

################################################################################
# 5. Axis isolation cases
################################################################################

for r in 1 2 3; do
    run_lc_vs_be "axis_lc_vs_1_be_long_warm_r${r}" "be-long" 1
    run_lc_vs_be "axis_lc_vs_4_be_chunked_warm_r${r}" "be-chunked" 4
done

log_note "Warm-clean Week 2 run finished: $(date)"

echo
echo "Done. Warm-clean results written to:"
echo "$OUT"

