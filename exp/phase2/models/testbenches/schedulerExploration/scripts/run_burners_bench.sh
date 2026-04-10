#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <launch_mode> <sync_mode> [burner_count] [bench_cores] [burner_mode]"
  echo "Example: $0 large per_iter"
  echo "Example: $0 small final 2 0-1 same"
  echo "Example: $0 small final 4 0-1 separate"
  exit 1
fi

LAUNCH_MODE="$1"
SYNC_MODE="$2"
BURNER_COUNT="${3:-0}"
BENCH_CORES="${4:-0-1}"
BURNER_MODE="${5:-same}"
THREADS="$6"

BENCH="./out/bench_cc"
RUN_NAME="bench_${SYNC_MODE}_${LAUNCH_MODE}_burn${BURNER_COUNT}_${BURNER_MODE}_bench$(echo "${BENCH_CORES}" | tr ',' '_' | tr '-' '_')_${THREADS}"
RUN_DIR="runs/${RUN_NAME}"

mkdir -p "${RUN_DIR}"
mkdir -p "${RUN_DIR}/strace"

echo "launch_mode=${LAUNCH_MODE}"            >  "${RUN_DIR}/config.txt"
echo "sync_mode=${SYNC_MODE}"                >> "${RUN_DIR}/config.txt"
echo "burner_count=${BURNER_COUNT}"          >> "${RUN_DIR}/config.txt"
echo "bench_cores=${BENCH_CORES}"            >> "${RUN_DIR}/config.txt"
echo "burner_mode=${BURNER_MODE}"            >> "${RUN_DIR}/config.txt"
echo "thread_count=${THREADS}"               >> "${RUN_DIR}/config.txt"

cp bench.cu kernels.cu kernels.cuh "${RUN_DIR}/" 2>/dev/null || true

BURNER_PIDS=()

expand_cpu_list() {
  python3 - "$1" <<'PY'
import sys

spec = sys.argv[1]
cpus = []

for part in spec.split(','):
    part = part.strip()
    if not part:
        continue
    if '-' in part:
        a, b = part.split('-', 1)
        a, b = int(a), int(b)
        step = 1 if a <= b else -1
        cpus.extend(range(a, b + step, step))
    else:
        cpus.append(int(part))

print(" ".join(map(str, cpus)))
PY
}

pick_separate_pool() {
  python3 - "$1" <<'PY'
import os
import sys

bench = set()
spec = sys.argv[1]
for part in spec.split(','):
    part = part.strip()
    if not part:
        continue
    if '-' in part:
        a, b = part.split('-', 1)
        a, b = int(a), int(b)
        step = 1 if a <= b else -1
        for x in range(a, b + step, step):
            bench.add(x)
    else:
        bench.add(int(part))

avail = list(range(os.cpu_count() or 1))
pool = [c for c in avail if c not in bench]

print(" ".join(map(str, pool)))
PY
}

start_burners() {
  local count="$1"
  local mode="$2"
  local bench_spec="$3"

  if [ "${count}" -le 0 ]; then
    return
  fi

  local cpu_pool
  if [ "${mode}" = "same" ]; then
    cpu_pool="$(expand_cpu_list "${bench_spec}")"
  elif [ "${mode}" = "separate" ]; then
    cpu_pool="$(pick_separate_pool "${bench_spec}")"
  else
    echo "Unknown burner_mode: ${mode}"
    exit 1
  fi

  if [ -z "${cpu_pool}" ]; then
    echo "No CPUs available for burner pool"
    exit 1
  fi

  echo "burner_cpu_pool=${cpu_pool}" >> "${RUN_DIR}/config.txt"

  read -r -a cpus <<< "${cpu_pool}"
  local ncpus="${#cpus[@]}"

  for ((i=0; i<count; i++)); do
    cpu="${cpus[$((i % ncpus))]}"
    taskset -c "${cpu}" sh -c 'while :; do :; done' &
    BURNER_PIDS+=("$!")
  done
}

cleanup() {
  for pid in "${BURNER_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT

start_burners "${BURNER_COUNT}" "${BURNER_MODE}" "${BENCH_CORES}"

echo "taskset -c ${BENCH_CORES} ${BENCH} ${LAUNCH_MODE} ${SYNC_MODE}" > "${RUN_DIR}/command.txt"
if [ "${BURNER_COUNT}" -gt 0 ]; then
  echo "burners=${BURNER_COUNT} mode=${BURNER_MODE}" >> "${RUN_DIR}/command.txt"
fi

strace -ff -ttt -T \
  -e trace=ioctl,futex,poll,ppoll,epoll_wait,nanosleep,clock_nanosleep,mmap,munmap,mprotect \
  -o "${RUN_DIR}/strace/trace" \
  taskset -c "${BENCH_CORES}" "${BENCH}" "${LAUNCH_MODE}" "${SYNC_MODE}" "${THREADS}" \
  > "${RUN_DIR}/stdout.txt" \
  2> "${RUN_DIR}/stderr.txt"

echo "done" > "${RUN_DIR}/status.txt"
