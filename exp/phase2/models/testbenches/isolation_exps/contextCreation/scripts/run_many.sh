#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <binary> <base_run_dir> <count>"
    exit 1
fi

BIN="$1"
BASE_DIR="$2"
COUNT="$3"

mkdir -p "$BASE_DIR"

for i in $(seq 1 "$COUNT"); do
    RUN_ID=$(printf "run_%03d" "$i")
    RUN_DIR="$BASE_DIR/$RUN_ID"
    mkdir -p "$RUN_DIR"

    STRACE_OUT="$RUN_DIR/trace"
    STDERR_OUT="$RUN_DIR/stderr.txt"
    STDOUT_OUT="$RUN_DIR/stdout.txt"


    echo "[$RUN_ID] running..."

    strace -ff -ttt -T \
      -e trace=ioctl,mmap,munmap,mprotect,futex,poll,ppoll,epoll_wait,clone,clone3 \
      -o "$STRACE_OUT" \
      "$BIN" \
      >"$STDOUT_OUT" \
      2>"$STDERR_OUT"

done

echo "All runs complete: $BASE_DIR"
