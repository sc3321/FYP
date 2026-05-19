#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./out/policyWorkload}"
OUT="${OUT:-./out/policy_three_way_shared_mem_simple}"

TRACE_SYSCALLS="${TRACE_SYSCALLS:-futex,poll,ppoll,epoll_wait,ioctl,nanosleep,clock_nanosleep,mmap,munmap,mprotect,write}"

LC_ITERS="${LC_ITERS:-1}"
BE_ITERS="${BE_ITERS:-1}"
CHUNKS="${CHUNKS:-1}"
LC_SLEEP_US="${LC_SLEEP_US:-0}"

rm -rf "$OUT"
mkdir -p "$OUT"/{strace,lc_events,be_long_events,be_chunked_events}

COMBINED="$OUT/combined.out.txt"
touch "$COMBINED"

echo "Policy three-way simple run started: $(date)" | tee -a "$COMBINED"
echo "Output: $OUT" | tee -a "$COMBINED"
echo "Binary: $BIN" | tee -a "$COMBINED"

cat > "$OUT/config.txt" <<EOF
BIN=$BIN
OUT=$OUT
TRACE_SYSCALLS=$TRACE_SYSCALLS
LC_ITERS=$LC_ITERS
BE_ITERS=$BE_ITERS
CHUNKS=$CHUNKS
LC_SLEEP_US=$LC_SLEEP_US
EOF

echo "[RUN] starting BE-long" | tee -a "$COMBINED"
GPU_PHASE_LOG_DIR="$OUT/be_long_events" \
strace -ff -ttt -T \
    -e trace="$TRACE_SYSCALLS" \
    -s 128 \
    -o "$OUT/strace/be_long.trace" \
    "$BIN" \
        --mode be-long \
        --iters "$BE_ITERS" \
        --chunks "$CHUNKS" \
    > "$OUT/be_long.stdout.txt" \
    2> "$OUT/be_long.stderr.txt" &
BE_LONG_PID=$!

echo "[RUN] starting BE-chunked" | tee -a "$COMBINED"
GPU_PHASE_LOG_DIR="$OUT/be_chunked_events" \
strace -ff -ttt -T \
    -e trace="$TRACE_SYSCALLS" \
    -s 128 \
    -o "$OUT/strace/be_chunked.trace" \
    "$BIN" \
        --mode be-chunked \
        --iters "$BE_ITERS" \
        --chunks "$CHUNKS" \
    > "$OUT/be_chunked.stdout.txt" \
    2> "$OUT/be_chunked.stderr.txt" &
BE_CHUNKED_PID=$!

sleep 0.2

echo "[RUN] starting LC" | tee -a "$COMBINED"
GPU_PHASE_LOG_DIR="$OUT/lc_events" \
strace -ff -ttt -T \
    -e trace="$TRACE_SYSCALLS" \
    -s 128 \
    -o "$OUT/strace/lc.trace" \
    "$BIN" \
        --mode lc \
        --iters "$LC_ITERS" \
        --chunks "$CHUNKS" \
        --sleep-us "$LC_SLEEP_US" \
    > "$OUT/lc.stdout.txt" \
    2> "$OUT/lc.stderr.txt" &
LC_PID=$!

echo "BE_LONG_PID=$BE_LONG_PID" | tee -a "$COMBINED"
echo "BE_CHUNKED_PID=$BE_CHUNKED_PID" | tee -a "$COMBINED"
echo "LC_PID=$LC_PID" | tee -a "$COMBINED"

status=0

wait "$LC_PID" || status=1
wait "$BE_LONG_PID" || status=1
wait "$BE_CHUNKED_PID" || status=1

{
    echo
    echo "===== LC STDERR ====="
    cat "$OUT/lc.stderr.txt" || true

    echo
    echo "===== BE LONG STDERR ====="
    cat "$OUT/be_long.stderr.txt" || true

    echo
    echo "===== BE CHUNKED STDERR ====="
    cat "$OUT/be_chunked.stderr.txt" || true

    echo
    echo "===== LC STDOUT ====="
    cat "$OUT/lc.stdout.txt" || true

    echo
    echo "===== BE LONG STDOUT ====="
    cat "$OUT/be_long.stdout.txt" || true

    echo
    echo "===== BE CHUNKED STDOUT ====="
    cat "$OUT/be_chunked.stdout.txt" || true
} >> "$COMBINED"

echo "Finished: $(date), status=$status" | tee -a "$COMBINED"

exit "$status"
