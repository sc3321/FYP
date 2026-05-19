#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./out/policyWorkload}"
OUT="${OUT:-./out/policy_three_way_shared_mem}"

TRACE_SYSCALLS="${TRACE_SYSCALLS:-futex,poll,ppoll,epoll_wait,ioctl,nanosleep,clock_nanosleep,mmap,munmap,mprotect,write}"

LC_ITERS="${LC_ITERS:-1}"
BE_ITERS="${BE_ITERS:-1}"
CHUNKS="${CHUNKS:-1}"
LC_SLEEP_US="${LC_SLEEP_US:-0}"

N="${N:-1024}"
LC_INNER="${LC_INNER:-1}"
BE_INNER="${BE_INNER:-1}"

rm -rf "$OUT"
mkdir -p "$OUT"/{strace,lc_events,be_long_events,be_chunked_events}

COMBINED="$OUT/combined.out.txt"
NOTES="$OUT/run_notes.txt"

touch "$COMBINED" "$NOTES"

log_note() {
    echo "$*" | tee -a "$NOTES"
}

cat > "$OUT/config.txt" <<EOF
BIN=$BIN
OUT=$OUT
TRACE_SYSCALLS=$TRACE_SYSCALLS
LC_ITERS=$LC_ITERS
BE_ITERS=$BE_ITERS
CHUNKS=$CHUNKS
LC_SLEEP_US=$LC_SLEEP_US
N=$N
LC_INNER=$LC_INNER
BE_INNER=$BE_INNER
EOF

log_note "Policy three-way shared-memory run started: $(date)"
log_note "Output: $OUT"
log_note "Binary: $BIN"
log_note "Combined output: $COMBINED"

log_note
log_note "======================================================================"
log_note "[RUN] three_way_lc_be_long_be_chunked"
log_note "LC + BE-long + BE-chunked concurrently"
log_note "======================================================================"

log_note "[START] BE-long"

GPU_PHASE_LOG_DIR="$OUT/be_long_events" \
strace -ff -ttt -T \
    -e trace="$TRACE_SYSCALLS" \
    -s 128 \
    -o "$OUT/strace/be_long.trace" \
    "$BIN" \
        --mode be-long \
        --iters "$BE_ITERS" \
        --chunks "$CHUNKS" \
        --n "$N" \
        --be-inner "$BE_INNER" \
    > "$OUT/be_long.stdout.txt" \
    2> "$OUT/be_long.stderr.txt" &

BE_LONG_PID=$!

log_note "[START] BE-chunked"

GPU_PHASE_LOG_DIR="$OUT/be_chunked_events" \
strace -ff -ttt -T \
    -e trace="$TRACE_SYSCALLS" \
    -s 128 \
    -o "$OUT/strace/be_chunked.trace" \
    "$BIN" \
        --mode be-chunked \
        --iters "$BE_ITERS" \
        --chunks "$CHUNKS" \
        --n "$N" \
        --be-inner "$BE_INNER" \
    > "$OUT/be_chunked.stdout.txt" \
    2> "$OUT/be_chunked.stderr.txt" &

BE_CHUNKED_PID=$!

sleep 0.2

log_note "[START] LC"

GPU_PHASE_LOG_DIR="$OUT/lc_events" \
strace -ff -ttt -T \
    -e trace="$TRACE_SYSCALLS" \
    -s 128 \
    -o "$OUT/strace/lc.trace" \
    "$BIN" \
        --mode lc \
        --iters "$LC_ITERS" \
        --chunks "$CHUNKS" \
        --n "$N" \
        --lc-inner "$LC_INNER" \
        --sleep-us "$LC_SLEEP_US" \
    > "$OUT/lc.stdout.txt" \
    2> "$OUT/lc.stderr.txt" &

LC_PID=$!

{
    echo "BE_LONG_PID=$BE_LONG_PID"
    echo "BE_CHUNKED_PID=$BE_CHUNKED_PID"
    echo "LC_PID=$LC_PID"
} | tee "$OUT/pids.txt"

status=0

log_note "[WAIT] LC"
if ! wait "$LC_PID"; then
    log_note "[ERROR] LC failed"
    status=1
fi

log_note "[WAIT] BE-long"
if ! wait "$BE_LONG_PID"; then
    log_note "[ERROR] BE-long failed"
    status=1
fi

log_note "[WAIT] BE-chunked"
if ! wait "$BE_CHUNKED_PID"; then
    log_note "[ERROR] BE-chunked failed"
    status=1
fi

{
    echo
    echo "======================================================================"
    echo "RUN NOTES"
    echo "======================================================================"
    cat "$NOTES" || true

    echo
    echo "======================================================================"
    echo "PIDS"
    echo "======================================================================"
    cat "$OUT/pids.txt" || true

    echo
    echo "======================================================================"
    echo "LC STDERR"
    echo "======================================================================"
    cat "$OUT/lc.stderr.txt" || true

    echo
    echo "======================================================================"
    echo "BE LONG STDERR"
    echo "======================================================================"
    cat "$OUT/be_long.stderr.txt" || true

    echo
    echo "======================================================================"
    echo "BE CHUNKED STDERR"
    echo "======================================================================"
    cat "$OUT/be_chunked.stderr.txt" || true

    echo
    echo "======================================================================"
    echo "LC STDOUT"
    echo "======================================================================"
    cat "$OUT/lc.stdout.txt" || true

    echo
    echo "======================================================================"
    echo "BE LONG STDOUT"
    echo "======================================================================"
    cat "$OUT/be_long.stdout.txt" || true

    echo
    echo "======================================================================"
    echo "BE CHUNKED STDOUT"
    echo "======================================================================"
    cat "$OUT/be_chunked.stdout.txt" || true
} > "$COMBINED"

log_note "Policy three-way shared-memory run finished: $(date)"
log_note "Status: $status"

echo
echo "Done. Results written to:"
echo "$OUT"
echo
echo "Combined output:"
echo "$COMBINED"

exit "$status"
