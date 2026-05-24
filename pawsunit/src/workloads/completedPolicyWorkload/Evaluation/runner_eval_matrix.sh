#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# runner_eval_matrix.sh
#
# Clean Phase 3 policy evaluation runner.
#
# Worker meaning:
#   LC_WORKERS is always 1.
#   LONG_BE_WORKERS is the number of independent BE long processes.
#   CHUNKED_BE_WORKERS is the number of independent BE chunked processes.
#   TOTAL_BE_WORKERS = LONG_BE_WORKERS + CHUNKED_BE_WORKERS.
#
# Named worker configs:
#   lc_only      -> LC only
#   long1        -> 1 long BE worker
#   long4        -> 4 long BE workers
#   chunked1     -> 1 chunked BE worker
#   chunked4     -> 4 chunked BE workers
#   mixed2       -> 1 long + 1 chunked = 2 total BE workers
#   mixed4       -> 2 long + 2 chunked = 4 total BE workers
#
# Recommended use:
#   chmod +x runner_eval_matrix.sh
#   ./runner_eval_matrix.sh
#
# Example targeted run:
#   MATRIX_MODE=delay_sweep CASE_FILTER=lc_vs_mixed WCFG_FILTER=mixed4 ./runner_eval_matrix.sh
###############################################################################

###############################################################################
# Paths
###############################################################################

BIN="${BIN:-../out/policyWorkload}"
OUT_ROOT="${OUT_ROOT:-./out_matrix}"

if [[ ! -x "$BIN" ]]; then
  echo "[ERROR] Binary not found or not executable: $BIN" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

###############################################################################
# Experiment defaults
###############################################################################

REPEATS="${REPEATS:-5}"

# Modes:
#   core          -> compact core result
#   workers       -> worker-scaling matrix
#   delay_sweep   -> proper-policy delay sweep
#   all           -> core + workers + delay_sweep
MATRIX_MODE="${MATRIX_MODE:-core}"

# Optional filters.
# Leave empty to run everything selected by MATRIX_MODE.
CASE_FILTER="${CASE_FILTER:-}"
WCFG_FILTER="${WCFG_FILTER:-}"
POLICY_FILTER="${POLICY_FILTER:-}"
DELAY_FILTER="${DELAY_FILTER:-}"

# Start order:
#   be_first is the stress case: BE is already active, then LC arrives.
#   lc_first is weaker: LC gets a head start.
#   simultaneous currently means BE workers are spawned first with no sleep, then LC.
START_ORDER="${START_ORDER:-be_first}"
STAGGER_SEC="${STAGGER_SEC:-0.05}"

# Workload parameters.
N="${N:-1048576}"
LC_ITERS="${LC_ITERS:-50}"
BE_ITERS="${BE_ITERS:-80}"
CHUNKS="${CHUNKS:-8}"
LC_INNER="${LC_INNER:-64}"
BE_INNER="${BE_INNER:-512}"
SLEEP_US="${SLEEP_US:-0}"

# Tracing.
TRACE="${TRACE:-1}"
STRACE_FLAGS="${STRACE_FLAGS:--ff -ttt -T -e trace=futex,poll,ppoll,epoll_wait,ioctl,nanosleep,clock_nanosleep,mmap,munmap,mprotect,write -s 128}"

# CUDA device selection.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

###############################################################################
# Helpers
###############################################################################

contains_filter() {
  local value="$1"
  local filter="$2"

  if [[ -z "$filter" ]]; then
    return 0
  fi

  for x in $filter; do
    if [[ "$x" == "$value" ]]; then
      return 0
    fi
  done

  return 1
}

sleep_stagger() {
  if [[ "$START_ORDER" != "simultaneous" ]]; then
    sleep "$STAGGER_SEC"
  fi
}

case_uses_lc() {
  case "$1" in
    lc_alone|lc_vs_be_long|lc_vs_be_chunked|lc_vs_mixed) return 0 ;;
    *) return 1 ;;
  esac
}

case_uses_be() {
  case "$1" in
    be_long_alone|be_chunked_alone|lc_vs_be_long|lc_vs_be_chunked|lc_vs_mixed) return 0 ;;
    *) return 1 ;;
  esac
}

is_alone_case() {
  case "$1" in
    lc_alone|be_long_alone|be_chunked_alone) return 0 ;;
    *) return 1 ;;
  esac
}

###############################################################################
# Worker config decoder
###############################################################################

decode_wcfg() {
  local wcfg="$1"

  LONG_BE_WORKERS=0
  CHUNKED_BE_WORKERS=0

  case "$wcfg" in
    lc_only)
      LONG_BE_WORKERS=0
      CHUNKED_BE_WORKERS=0
      ;;
    long1)
      LONG_BE_WORKERS=1
      CHUNKED_BE_WORKERS=0
      ;;
    long4)
      LONG_BE_WORKERS=4
      CHUNKED_BE_WORKERS=0
      ;;
    chunked1)
      LONG_BE_WORKERS=0
      CHUNKED_BE_WORKERS=1
      ;;
    chunked4)
      LONG_BE_WORKERS=0
      CHUNKED_BE_WORKERS=4
      ;;
    mixed2)
      LONG_BE_WORKERS=1
      CHUNKED_BE_WORKERS=1
      ;;
    mixed4)
      LONG_BE_WORKERS=2
      CHUNKED_BE_WORKERS=2
      ;;
    *)
      echo "[ERROR] Unknown worker config: $wcfg" >&2
      exit 1
      ;;
  esac

  TOTAL_BE_WORKERS=$((LONG_BE_WORKERS + CHUNKED_BE_WORKERS))
}

valid_wcfg_for_case() {
  local case_name="$1"
  local wcfg="$2"

  case "$case_name" in
    lc_alone)
      [[ "$wcfg" == "lc_only" ]]
      ;;
    be_long_alone|lc_vs_be_long)
      [[ "$wcfg" == long* ]]
      ;;
    be_chunked_alone|lc_vs_be_chunked)
      [[ "$wcfg" == chunked* ]]
      ;;
    lc_vs_mixed)
      [[ "$wcfg" == mixed* ]]
      ;;
    *)
      return 1
      ;;
  esac
}

###############################################################################
# Process launcher
###############################################################################

start_proc() {
  local run_dir="$1"
  local run_id="$2"
  local policy="$3"
  local delay_us="$4"
  local proc_name="$5"
  local mode="$6"
  local worker_idx="$7"

  local event_dir="$run_dir/${proc_name}_events"
  local trace_dir="$run_dir/strace/${proc_name}"

  mkdir -p "$event_dir" "$trace_dir"

  local stdout_file="$run_dir/${proc_name}.stdout"
  local stderr_file="$run_dir/${proc_name}.stderr"

  local -a cmd

  case "$mode" in
    lc)
      cmd=("$BIN"
           --class LC
           --mode lc
           --policy "$policy"
           --iters "$LC_ITERS"
           --n "$N"
           --lc-inner "$LC_INNER"
           --sleep-us "$SLEEP_US")
      ;;
    be_long)
      cmd=("$BIN"
           --class BE
           --mode be-long
           --policy "$policy"
           --iters "$BE_ITERS"
           --n "$N"
           --be-inner "$BE_INNER"
           --chunks "$CHUNKS"
           --sleep-us "$SLEEP_US")
      ;;
    be_chunked)
      cmd=("$BIN"
           --class BE
           --mode be-chunked
           --policy "$policy"
           --iters "$BE_ITERS"
           --n "$N"
           --be-inner "$BE_INNER"
           --chunks "$CHUNKS"
           --sleep-us "$SLEEP_US")
      ;;
    *)
      echo "[ERROR] Unknown mode: $mode" >&2
      exit 1
      ;;
  esac

  echo "[START] policy=$policy delay_us=$delay_us proc=$proc_name mode=$mode worker=$worker_idx" >&2

  local -a env_vars
  env_vars=(
    "GPU_PHASE_LOG_DIR=$event_dir"
    "GPU_PHASE_RUN_ID=$run_id"
    "BE_DELAY_US=$delay_us"
  )

  if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    env_vars+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
  fi

  if [[ "$TRACE" == "1" ]]; then
    env "${env_vars[@]}" \
      strace $STRACE_FLAGS -o "$trace_dir/trace" "${cmd[@]}" \
      > "$stdout_file" 2> "$stderr_file" &
  else
    env "${env_vars[@]}" \
      "${cmd[@]}" \
      > "$stdout_file" 2> "$stderr_file" &
  fi

  LAST_PID=$!
}

###############################################################################
# Single run
###############################################################################

run_one() {
  local case_name="$1"
  local wcfg="$2"
  local policy="$3"
  local delay_us="$4"
  local repeat="$5"

  decode_wcfg "$wcfg"

  local run_id="${case_name}_${wcfg}_${policy}_delay${delay_us}_r${repeat}"
  local run_dir="$OUT_ROOT/$case_name/$wcfg/$policy/delay_${delay_us}/r${repeat}"

  rm -rf "$run_dir"
  mkdir -p "$run_dir"

  cat > "$run_dir/config.txt" <<EOF
CASE=$case_name
WORKER_CONFIG=$wcfg
REPEAT=$repeat
POLICY=$policy
BE_DELAY_US=$delay_us

LC_WORKERS=1
LONG_BE_WORKERS=$LONG_BE_WORKERS
CHUNKED_BE_WORKERS=$CHUNKED_BE_WORKERS
TOTAL_BE_WORKERS=$TOTAL_BE_WORKERS

START_ORDER=$START_ORDER
STAGGER_SEC=$STAGGER_SEC

N=$N
LC_ITERS=$LC_ITERS
BE_ITERS=$BE_ITERS
CHUNKS=$CHUNKS
LC_INNER=$LC_INNER
BE_INNER=$BE_INNER
SLEEP_US=$SLEEP_US

TRACE=$TRACE
BIN=$BIN
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
GPU_PHASE_RUN_ID=$run_id
EOF

  echo
  echo "================================================================================"
  echo "[RUN] case=$case_name wcfg=$wcfg policy=$policy delay_us=$delay_us repeat=$repeat"
  echo "      long_be=$LONG_BE_WORKERS chunked_be=$CHUNKED_BE_WORKERS total_be=$TOTAL_BE_WORKERS"
  echo "      output=$run_dir"
  echo "================================================================================"

  local -a pids=()

  launch_lc() {
    start_proc "$run_dir" "$run_id" "$policy" "$delay_us" "lc" "lc" 0
    pids+=("$LAST_PID")
  }

  launch_long_be_workers() {
    local idx_start="$1"
    local idx="$idx_start"

    for i in $(seq 1 "$LONG_BE_WORKERS"); do
      start_proc "$run_dir" "$run_id" "$policy" "$delay_us" "be${idx}" "be_long" "$idx"
      pids+=("$LAST_PID")
      idx=$((idx + 1))
    done

    NEXT_IDX="$idx"
  }

  launch_chunked_be_workers() {
    local idx_start="$1"
    local idx="$idx_start"

    for i in $(seq 1 "$CHUNKED_BE_WORKERS"); do
      start_proc "$run_dir" "$run_id" "$policy" "$delay_us" "be${idx}" "be_chunked" "$idx"
      pids+=("$LAST_PID")
      idx=$((idx + 1))
    done

    NEXT_IDX="$idx"
  }

  launch_be_workers() {
    launch_long_be_workers 1
    launch_chunked_be_workers "$NEXT_IDX"
  }

  case "$case_name" in
    lc_alone)
      launch_lc
      ;;

    be_long_alone)
      launch_long_be_workers 1
      ;;

    be_chunked_alone)
      launch_chunked_be_workers 1
      ;;

    lc_vs_be_long|lc_vs_be_chunked|lc_vs_mixed)
      if [[ "$START_ORDER" == "lc_first" ]]; then
        launch_lc
        sleep_stagger
        launch_be_workers
      else
        launch_be_workers
        sleep_stagger
        launch_lc
      fi
      ;;

    *)
      echo "[ERROR] Unknown case: $case_name" >&2
      exit 1
      ;;
  esac

  printf "%s\n" "${pids[@]}" > "$run_dir/pids.txt"

  local fail=0

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[WARN] Process failed: pid=$pid case=$case_name wcfg=$wcfg policy=$policy delay=$delay_us repeat=$repeat" >&2
      fail=1
    fi
  done

  if [[ "$fail" == "0" ]]; then
    echo "[DONE] case=$case_name wcfg=$wcfg policy=$policy delay_us=$delay_us repeat=$repeat"
  else
    echo "[DONE_WITH_FAILURE] case=$case_name wcfg=$wcfg policy=$policy delay_us=$delay_us repeat=$repeat"
  fi
}

###############################################################################
# Matrix definitions
###############################################################################

CORE_CASES=(
  "lc_alone"
  "be_long_alone"
  "be_chunked_alone"
  "lc_vs_be_long"
  "lc_vs_be_chunked"
  "lc_vs_mixed"
)

CORE_WCFGS=(
  "lc_only"
  "long1"
  "chunked1"
  "mixed2"
)

WORKER_CASES=(
  "be_long_alone"
  "be_chunked_alone"
  "lc_vs_be_long"
  "lc_vs_be_chunked"
  "lc_vs_mixed"
)

WORKER_WCFGS=(
  "long1"
  "long4"
  "chunked1"
  "chunked4"
  "mixed2"
  "mixed4"
)

DELAY_CASES=(
  "lc_vs_be_long"
  "lc_vs_be_chunked"
  "lc_vs_mixed"
)

DELAY_WCFGS=(
  "long4"
  "chunked4"
  "mixed4"
)

CORE_POLICIES=(
  "none"
  "naive"
  "proper"
)

# Baselines should only use none.
BASELINE_POLICIES=(
  "none"
)

DELAY_POLICIES=(
  "proper"
)

CORE_DELAYS=(
  "50"
)

DELAY_SWEEP_DELAYS=(
  "5000"
  "10000"
  "25000"
  "50000"
  "72000"
  "87500"
  "100000"
)

###############################################################################
# Matrix executor
###############################################################################

run_matrix_entry_set() {
  local matrix_name="$1"
  shift

  local -n cases_ref="$1"
  local -n wcfgs_ref="$2"
  local -n policies_ref="$3"
  local -n delays_ref="$4"

  echo
  echo "################################################################################"
  echo "[MATRIX] $matrix_name"
  echo "################################################################################"

  for repeat in $(seq 1 "$REPEATS"); do
    for case_name in "${cases_ref[@]}"; do
      contains_filter "$case_name" "$CASE_FILTER" || continue

      for wcfg in "${wcfgs_ref[@]}"; do
        contains_filter "$wcfg" "$WCFG_FILTER" || continue
        valid_wcfg_for_case "$case_name" "$wcfg" || continue

        for policy in "${policies_ref[@]}"; do
          contains_filter "$policy" "$POLICY_FILTER" || continue

          # Alone cases only run with policy=none.
          if is_alone_case "$case_name" && [[ "$policy" != "none" ]]; then
            continue
          fi

          for delay_us in "${delays_ref[@]}"; do
            contains_filter "$delay_us" "$DELAY_FILTER" || continue

            # For policy=none, delay should be recorded as 0 for clarity.
            local actual_delay="$delay_us"
            if [[ "$policy" == "none" ]]; then
              actual_delay="0"
            fi

            run_one "$case_name" "$wcfg" "$policy" "$actual_delay" "$repeat"
          done
        done
      done
    done
  done
}

###############################################################################
# Main
###############################################################################

echo "[CONFIG] BIN=$BIN"
echo "[CONFIG] OUT_ROOT=$OUT_ROOT"
echo "[CONFIG] MATRIX_MODE=$MATRIX_MODE"
echo "[CONFIG] REPEATS=$REPEATS"
echo "[CONFIG] START_ORDER=$START_ORDER STAGGER_SEC=$STAGGER_SEC"
echo "[CONFIG] N=$N LC_ITERS=$LC_ITERS BE_ITERS=$BE_ITERS CHUNKS=$CHUNKS"
echo "[CONFIG] LC_INNER=$LC_INNER BE_INNER=$BE_INNER SLEEP_US=$SLEEP_US"
echo "[CONFIG] TRACE=$TRACE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[CONFIG] CASE_FILTER=$CASE_FILTER"
echo "[CONFIG] WCFG_FILTER=$WCFG_FILTER"
echo "[CONFIG] POLICY_FILTER=$POLICY_FILTER"
echo "[CONFIG] DELAY_FILTER=$DELAY_FILTER"

case "$MATRIX_MODE" in
  core)
    run_matrix_entry_set "core" CORE_CASES CORE_WCFGS CORE_POLICIES CORE_DELAYS
    ;;

  workers)
    run_matrix_entry_set "workers" WORKER_CASES WORKER_WCFGS CORE_POLICIES CORE_DELAYS
    ;;

  delay_sweep)
    run_matrix_entry_set "delay_sweep" DELAY_CASES DELAY_WCFGS DELAY_POLICIES DELAY_SWEEP_DELAYS
    ;;

  all)
    run_matrix_entry_set "core" CORE_CASES CORE_WCFGS CORE_POLICIES CORE_DELAYS
    run_matrix_entry_set "workers" WORKER_CASES WORKER_WCFGS CORE_POLICIES CORE_DELAYS
    run_matrix_entry_set "delay_sweep" DELAY_CASES DELAY_WCFGS DELAY_POLICIES DELAY_SWEEP_DELAYS
    ;;

  *)
    echo "[ERROR] Unknown MATRIX_MODE=$MATRIX_MODE" >&2
    echo "        Use: core, workers, delay_sweep, all" >&2
    exit 1
    ;;
esac

echo
echo "[ALL DONE] results under: $OUT_ROOT"
