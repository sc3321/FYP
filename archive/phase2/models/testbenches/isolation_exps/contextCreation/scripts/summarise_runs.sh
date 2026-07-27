#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <base_run_dir>"
    exit 1
fi

BASE_DIR="$1"

printf "run_id,ioctl,mmap,munmap,mprotect,futex,poll,ppoll,epoll_wait,clone,clone3\n"

for RUN_DIR in "$BASE_DIR"/run_*; do
    [ -d "$RUN_DIR" ] || continue

    RUN_ID=$(basename "$RUN_DIR")

    TRACE_FILES=$(ls "$RUN_DIR"/trace* 2>/dev/null || true)

    ioctl_count=0
    mmap_count=0
    munmap_count=0
    mprotect_count=0
    futex_count=0
    poll_count=0
    ppoll_count=0
    epoll_wait_count=0
    clone_count=0
    clone3_count=0

    if [ -n "$TRACE_FILES" ]; then
        ioctl_count=$(cat $TRACE_FILES | grep -c '^.*ioctl(' || true)
        mmap_count=$(cat $TRACE_FILES | grep -c '^.*mmap(' || true)
        munmap_count=$(cat $TRACE_FILES | grep -c '^.*munmap(' || true)
        mprotect_count=$(cat $TRACE_FILES | grep -c '^.*mprotect(' || true)
        futex_count=$(cat $TRACE_FILES | grep -c '^.*futex(' || true)
        poll_count=$(cat $TRACE_FILES | grep -c '^.*poll(' || true)
        ppoll_count=$(cat $TRACE_FILES | grep -c '^.*ppoll(' || true)
        epoll_wait_count=$(cat $TRACE_FILES | grep -c '^.*epoll_wait(' || true)
        clone_count=$(cat $TRACE_FILES | grep -c '^.*clone(' || true)
        clone3_count=$(cat $TRACE_FILES | grep -c '^.*clone3(' || true)
    fi

    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$RUN_ID" \
        "$ioctl_count" \
        "$mmap_count" \
        "$munmap_count" \
        "$mprotect_count" \
        "$futex_count" \
        "$poll_count" \
        "$ppoll_count" \
        "$epoll_wait_count" \
        "$clone_count" \
        "$clone3_count"
done
