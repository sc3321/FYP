#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# runner_eval.sh
#
# Run the Phase 3 policy evaluation matrix.
#
# Expected layout:
#   project_root/
#     out/policyWorkload
#     parserPolicy.py
#     Evaluation/
#       runner_eval.sh
#       analyze_eval.py
#       out/
#
# Typical use from inside Evaluation/:
#   chmod +x runner_eval.sh
#   ./runner_eval.sh
#
# Override anything with environment variables, e.g.:
#   CASES="lc_vs_mixed" POLICIES="none naive proper" REPEATS=5 ./runner_eval.sh
###############################################################################

# Binary and output paths.
BIN="${BIN:-../out/policyWorkload}"
OUT_ROOT="${OUT_ROOT:-./out}"

# Cases:
#   lc_alone
#   be_long_alone
#   be_chunked_alone
#   lc_vs_be_long
#   lc_vs_be_chunked
#   lc_vs_mixed
CASES="${CASES:-lc_alone be_long_alone be_chunked_alone lc_vs_be_long lc_vs_be_chunked lc_vs_mixed}"

# Policies understood by the test binary:
#   none
#   naive
#   proper
POLICIES="${POLICIES:-none naive proper}"

# Number of repeats.
REPEATS="${REPEATS:-5}"

# If 1, standalone cases run only with policy=none because policy has no LC/BE
# interaction to act on there. Set to 0 if you explicitly want all policies.
BASELINE_ONLY_ALONE="${BASELINE_ONLY_ALONE:-1}"

# Start order for interference cases:
#   lc_first       LC starts first, then BE starts after STAGGER_SEC
#   be_first       BE starts first, then LC starts after STAGGER_SEC
#   simultaneous   no intentional delay
START_ORDER="${START_ORDER:-lc_first}"
STAGGER_SEC="${STAGGER_SEC:-0.05}"

# Workload parameters. These match the stronger condition you have been using.
N="${N:-1048576}"
LC_ITERS="${LC_ITERS:-50}"
BE_ITERS="${BE_ITERS:-80}"
CHUNKS="${CHUNKS:-8}"
LC_INNER="${LC_INNER:-64}"
BE_INNER="${BE_INNER:-512}"
SLEEP_US="${SLEEP_US:-0}"

# Worker controls.
# For long-only/chunked-only cases:
BE_WORKERS="${BE_WORKERS:-1}"
# For mixed case:
BE_LONG_WORKERS="${BE_LONG_WORKERS:-1}"
BE_CHUNKED_WORKERS="${BE_CHUNKED_WORKERS:-1}"

# Optional policy delay knob. Your policyManager can read this via getenv().
BE_DELAY_US="${BE_DELAY_US:-50}"

# Tracing. TRACE=1 enables strace collection for every process.
TRACE="${TRACE:-1}"
STRACE_FLAGS="${STRACE_FLAGS:--ff -ttt -T -e trace=futex,poll,ppoll,epoll_wait,ioctl,nanosleep,clock_nanosleep,mmap,munmap,mprotect,write -s 128}"

# CUDA device selection, if needed.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

if [[ ! -x "$BIN" ]]; then
  echo "[ERROR] Binary not executable or not found: $BIN" >&2
  echo "        Set BIN=/path/to/policyWorkload" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

is_alone_case() {
  case "$1" in
    lc_alone|be_long_alone|be_chunked_alone) return 0 ;;
    *) return 1 ;;
  esac
}

policy_list_for_case() {
  local case_name="$1"
  if is_alone_case "$case_name" && [[ "$BASELINE_ONLY_ALONE" == "1" ]]; then
    echo "none"
  else
    echo "$POLICIES"
  fi
}

sleep_stagger() {
  if [[ "$START_ORDER" != "simultaneous" ]]; then
    sleep "$STAGGER_SEC"
  fi
}

start_proc() {
  local run_dir="$1"
  local policy="$2"
  local proc_name="$3"
  local mode="$4"
  local worker_idx="$5"

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
      echo "[ERROR] unknown mode: $mode" >&2
      exit 1
      ;;
  esac

  echo "[START] policy=$policy proc=$proc_name mode=$mode worker=$worker_idx" >&2 >&2

  local -a env_vars
  env_vars=(GPU_PHASE_LOG_DIR="$event_dir" BE_DELAY_US="$BE_DELAY_US")

  if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    env_vars+=(CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
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

run_case_policy_repeat() {
  local case_name="$1"
  local policy="$2"
  local repeat="$3"

  local run_root="$OUT_ROOT/${case_name}_r${repeat}"
  local run_dir="$run_root/$policy"

  rm -rf "$run_dir"
  mkdir -p "$run_dir"

  cat > "$run_dir/config.txt" <<EOF
CASE=$case_name
POLICY=$policy
REPEAT=$repeat
START_ORDER=$START_ORDER
STAGGER_SEC=$STAGGER_SEC
N=$N
LC_ITERS=$LC_ITERS
BE_ITERS=$BE_ITERS
CHUNKS=$CHUNKS
LC_INNER=$LC_INNER
BE_INNER=$BE_INNER
SLEEP_US=$SLEEP_US
BE_WORKERS=$BE_WORKERS
BE_LONG_WORKERS=$BE_LONG_WORKERS
BE_CHUNKED_WORKERS=$BE_CHUNKED_WORKERS
BE_DELAY_US=$BE_DELAY_US
TRACE=$TRACE
BIN=$BIN
EOF

  echo
  echo "======================================================================"
  echo "[RUN] case=$case_name policy=$policy repeat=$repeat"
  echo "      output=$run_dir"
  echo "======================================================================"

  local -a pids=()

  case "$case_name" in
    lc_alone)
      start_proc "$run_dir" "$policy" "lc" "lc" 0

      pids+=("$LAST_PID")
      ;;

    be_long_alone)
      for i in $(seq 1 "$BE_WORKERS"); do
        start_proc "$run_dir" "$policy" "be${i}" "be_long" "$i"

        pids+=("$LAST_PID")
      done
      ;;

    be_chunked_alone)
      for i in $(seq 1 "$BE_WORKERS"); do
        start_proc "$run_dir" "$policy" "be${i}" "be_chunked" "$i"

        pids+=("$LAST_PID")
      done
      ;;

    lc_vs_be_long)
      if [[ "$START_ORDER" == "lc_first" ]]; then
        start_proc "$run_dir" "$policy" "lc" "lc" 0

        pids+=("$LAST_PID")
        sleep_stagger
        for i in $(seq 1 "$BE_WORKERS"); do
          start_proc "$run_dir" "$policy" "be${i}" "be_long" "$i"

          pids+=("$LAST_PID")
        done
      else
        for i in $(seq 1 "$BE_WORKERS"); do
          start_proc "$run_dir" "$policy" "be${i}" "be_long" "$i"

          pids+=("$LAST_PID")
        done
        sleep_stagger
        start_proc "$run_dir" "$policy" "lc" "lc" 0

        pids+=("$LAST_PID")
      fi
      ;;

    lc_vs_be_chunked)
      if [[ "$START_ORDER" == "lc_first" ]]; then
        start_proc "$run_dir" "$policy" "lc" "lc" 0

        pids+=("$LAST_PID")
        sleep_stagger
        for i in $(seq 1 "$BE_WORKERS"); do
          start_proc "$run_dir" "$policy" "be${i}" "be_chunked" "$i"

          pids+=("$LAST_PID")
        done
      else
        for i in $(seq 1 "$BE_WORKERS"); do
          start_proc "$run_dir" "$policy" "be${i}" "be_chunked" "$i"

          pids+=("$LAST_PID")
        done
        sleep_stagger
        start_proc "$run_dir" "$policy" "lc" "lc" 0

        pids+=("$LAST_PID")
      fi
      ;;

    lc_vs_mixed)
      if [[ "$START_ORDER" == "lc_first" ]]; then
        start_proc "$run_dir" "$policy" "lc" "lc" 0

        pids+=("$LAST_PID")
        sleep_stagger
        local idx=1
        for i in $(seq 1 "$BE_LONG_WORKERS"); do
          start_proc "$run_dir" "$policy" "be${idx}" "be_long" "$idx"

          pids+=("$LAST_PID")
          idx=$((idx + 1))
        done
        for i in $(seq 1 "$BE_CHUNKED_WORKERS"); do
          start_proc "$run_dir" "$policy" "be${idx}" "be_chunked" "$idx"

          pids+=("$LAST_PID")
          idx=$((idx + 1))
        done
      else
        local idx=1
        for i in $(seq 1 "$BE_LONG_WORKERS"); do
          start_proc "$run_dir" "$policy" "be${idx}" "be_long" "$idx"

          pids+=("$LAST_PID")
          idx=$((idx + 1))
        done
        for i in $(seq 1 "$BE_CHUNKED_WORKERS"); do
          start_proc "$run_dir" "$policy" "be${idx}" "be_chunked" "$idx"

          pids+=("$LAST_PID")
          idx=$((idx + 1))
        done
        sleep_stagger
        start_proc "$run_dir" "$policy" "lc" "lc" 0

        pids+=("$LAST_PID")
      fi
      ;;

    *)
      echo "[ERROR] unknown case: $case_name" >&2
      exit 1
      ;;
  esac

  printf "%s\n" "${pids[@]}" > "$run_dir/pids.txt"

  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[WARN] process failed: pid=$pid case=$case_name policy=$policy repeat=$repeat" >&2
      fail=1
    fi
  done

  if [[ "$fail" == "0" ]]; then
    echo "[DONE] case=$case_name policy=$policy repeat=$repeat"
  else
    echo "[DONE_WITH_FAILURE] case=$case_name policy=$policy repeat=$repeat"
  fi
}

echo "[CONFIG] OUT_ROOT=$OUT_ROOT"
echo "[CONFIG] CASES=$CASES"
echo "[CONFIG] POLICIES=$POLICIES"
echo "[CONFIG] REPEATS=$REPEATS"
echo "[CONFIG] START_ORDER=$START_ORDER STAGGER_SEC=$STAGGER_SEC"
echo "[CONFIG] N=$N LC_ITERS=$LC_ITERS BE_ITERS=$BE_ITERS CHUNKS=$CHUNKS LC_INNER=$LC_INNER BE_INNER=$BE_INNER"
echo "[CONFIG] BE_WORKERS=$BE_WORKERS BE_LONG_WORKERS=$BE_LONG_WORKERS BE_CHUNKED_WORKERS=$BE_CHUNKED_WORKERS"
echo "[CONFIG] BE_DELAY_US=$BE_DELAY_US TRACE=$TRACE"

for r in $(seq 1 "$REPEATS"); do
  for case_name in $CASES; do
    for policy in $(policy_list_for_case "$case_name"); do
      run_case_policy_repeat "$case_name" "$policy" "$r"
    done
  done
done

echo
echo "[ALL DONE] results under: $OUT_ROOT"

