#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <launch_mode> <sync_mode> [stress_threads]"
  echo "Example: $0 large per_iter"
  echo "Example: $0 small final 8"
  exit 1
fi

LAUNCH_MODE="$1"
SYNC_MODE="$2"
STRESS_THREADS="${3:-0}"

BENCH="./out/bench_cc"
RUN_NAME="bench_${SYNC_MODE}_${LAUNCH_MODE}_stress${STRESS_THREADS}"
RUN_DIR="runs/${RUN_NAME}"

mkdir -p "${RUN_DIR}"
mkdir -p "${RUN_DIR}/strace"

echo "launch_mode=${LAUNCH_MODE}"   >  "${RUN_DIR}/config.txt"
echo "sync_mode=${SYNC_MODE}"       >> "${RUN_DIR}/config.txt"
echo "stress_threads=${STRESS_THREADS}" >> "${RUN_DIR}/config.txt"

echo "${BENCH} ${LAUNCH_MODE} ${SYNC_MODE}" > "${RUN_DIR}/command.txt"
if [ "${STRESS_THREADS}" -gt 0 ]; then
  echo "stress -c ${STRESS_THREADS}" >> "${RUN_DIR}/command.txt"
fi

cp bench.cu kernels.cu kernels.cuh "${RUN_DIR}/" 2>/dev/null || true

STRESS_PID=""
cleanup() {
  if [ -n "${STRESS_PID}" ]; then
    kill "${STRESS_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [ "${STRESS_THREADS}" -gt 0 ]; then
  stress -c "${STRESS_THREADS}" &
  STRESS_PID=$!
fi

strace -ff -ttt -T \
  -e trace=ioctl,futex,poll,ppoll,epoll_wait,nanosleep,clock_nanosleep,mmap,munmap,mprotect \
  -o "${RUN_DIR}/strace/trace" \
  "${BENCH}" "${LAUNCH_MODE}" "${SYNC_MODE}" \
  > "${RUN_DIR}/stdout.txt" \
  2> "${RUN_DIR}/stderr.txt"

perf sched timehist \
  "${BENCH}" "${LAUNCH_MODE}" "${SYNC_MODE}" \
  > "${RUN_DIR}/sched.txt" \
  2>> "${RUN_DIR}/stderr.txt" || true

echo "done" > "${RUN_DIR}/status.txt"
