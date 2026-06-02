#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Repeated llama.cpp phase matrix runner
#
# Default N_RUNS=25 is calibrated for N_LC=50 inside each matrix run, which
# gives ~17 hours total wallclock. Use 20 for an overnight run, 25-30 if you
# can leave it longer. For very quick smoke-test, pass 3 on the command line.
#
# Each matrix run produces 8 cases, with warmup before each timed measurement.
# All cold-start artefacts (model load, JIT, CUDA context, KV cache prefix)
# should be cleared by the warmup phase before measured requests begin.
# ==============================================================================

ROOT="/home/sc3321/FYP/llama_validation"
RUNNER="${ROOT}/scripts/run_llama_phase_matrix.sh"
ANALYSER="${ROOT}/scripts/analyse_llama_phase_matrix.py"

N_RUNS="${1:-25}"

MASTER_OUT="${ROOT}/runs/llama_phase_repeated_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${MASTER_OUT}/logs"

mkdir -p "$MASTER_OUT" "$LOG_DIR"

cd "$ROOT"

echo "======================================================================"
echo "Repeated llama phase matrix experiment"
echo "======================================================================"
echo "Root:        $ROOT"
echo "Runner:      $RUNNER"
echo "Analyser:    $ANALYSER"
echo "N_RUNS:      $N_RUNS"
echo "MASTER_OUT:  $MASTER_OUT"
echo "Start time:  $(date)"
echo "======================================================================"

if [[ ! -x "$RUNNER" ]]; then
  echo "ERROR: runner script not found or not executable: $RUNNER" >&2
  echo "Run: chmod +x $RUNNER" >&2
  exit 1
fi

if [[ ! -f "$ANALYSER" ]]; then
  echo "ERROR: analysis script not found: $ANALYSER" >&2
  exit 1
fi

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

kill_old_servers() {
  pkill -u "$USER" -f llama-server 2>/dev/null || true
  rm -f /dev/shm/gpuphase_gpu0 2>/dev/null || true
  sleep 1
}

latest_matrix_dir() {
  ls -td "${ROOT}"/runs/llama_phase_matrix_* 2>/dev/null | head -1 || true
}

validate_run_dir() {
  local run_dir="$1"

  if [[ ! -d "$run_dir" ]]; then
    echo "ERROR: run dir does not exist: $run_dir" >&2
    return 1
  fi

  if [[ ! -f "$run_dir/summary.csv" ]]; then
    echo "ERROR: missing summary.csv in $run_dir" >&2
    return 1
  fi

  local required_cases=(
    "caseA_lc_alone_none"
    "caseB_be_long_alone_none"
    "caseC_lc_be_long_none"
    "caseD_lc_be_long_policy"
    "caseE_lc_be_short_none"
    "caseF_lc_be_short_policy"
    "caseG_lc_first_be_long_none"
    "caseH_lc_first_be_long_policy"
  )

  for case_name in "${required_cases[@]}"; do
    if [[ ! -d "$run_dir/$case_name" ]]; then
      echo "ERROR: missing case directory $case_name in $run_dir" >&2
      return 1
    fi
  done

  return 0
}

count_events() {
  local run_dir="$1"
  grep -R "Event type =" "$run_dir" 2>/dev/null | wc -l || true
}

write_master_metadata() {
  cat > "${MASTER_OUT}/metadata.txt" <<EOF
master_out=${MASTER_OUT}
root=${ROOT}
runner=${RUNNER}
analyser=${ANALYSER}
n_runs_requested=${N_RUNS}
start_time=$(date)
user=${USER}
hostname=$(hostname)
pwd=$(pwd)
EOF
}

write_master_metadata

mkdir -p "${MASTER_OUT}/scripts_used"
cp "$RUNNER" "${MASTER_OUT}/scripts_used/run_llama_phase_matrix.sh"
cp "$ANALYSER" "${MASTER_OUT}/scripts_used/analyse_llama_phase_matrix.py"

{
  echo "===== date ====="
  date
  echo
  echo "===== hostname ====="
  hostname
  echo
  echo "===== git status llama.cpp ====="
  if [[ -d "${ROOT}/llama.cpp/.git" ]]; then
    git -C "${ROOT}/llama.cpp" rev-parse HEAD || true
    git -C "${ROOT}/llama.cpp" status --short || true
  else
    echo "No git repo found at ${ROOT}/llama.cpp"
  fi
  echo
  echo "===== nvidia-smi before ====="
  nvidia-smi || true
  echo
  echo "===== relevant env ====="
  env | grep -E "GPU_PHASE|POLICY|BE_DELAY|CUDA|LLAMA|GGML" || true
} > "${MASTER_OUT}/environment_before.txt" 2>&1

STATUS_CSV="${MASTER_OUT}/run_status.csv"

cat > "$STATUS_CSV" <<EOF
run_index,status,source_run_dir,dest_run_dir,event_count,start_time,end_time,duration_s
EOF

success_count=0
fail_count=0

for run_idx in $(seq 1 "$N_RUNS"); do
  run_label=$(printf "run_%03d" "$run_idx")
  run_log="${LOG_DIR}/${run_label}.log"
  run_status="FAILED"
  source_run_dir=""
  dest_run_dir="${MASTER_OUT}/${run_label}"
  event_count=0

  start_epoch=$(date +%s)
  start_time="$(timestamp)"

  echo
  echo "======================================================================"
  echo "[$start_time] Starting $run_label / $N_RUNS"
  echo "======================================================================"

  {
    echo "======================================================================"
    echo "Run:        $run_label"
    echo "Start time: $start_time"
    echo "Root:       $ROOT"
    echo "Master:     $MASTER_OUT"
    echo "======================================================================"
    echo

    echo "Killing old servers and clearing shared memory..."
    kill_old_servers

    before_latest="$(latest_matrix_dir)"
    echo "Latest matrix before run: ${before_latest:-<none>}"
    echo

    echo "nvidia-smi before run:"
    nvidia-smi || true
    echo

    echo "Running matrix..."
    "$RUNNER"

    echo
    echo "Matrix script completed."

    after_latest="$(latest_matrix_dir)"
    echo "Latest matrix after run: ${after_latest:-<none>}"

    if [[ -z "$after_latest" ]]; then
      echo "ERROR: could not locate generated runs/llama_phase_matrix_* directory" >&2
      exit 1
    fi

    if [[ "$after_latest" == "$before_latest" ]]; then
      echo "WARNING: latest matrix directory did not change."
      echo "Proceeding with latest directory anyway: $after_latest"
    fi

    source_run_dir="$after_latest"

    echo
    echo "Running per-run analysis on: $source_run_dir"
    python3 "$ANALYSER" "$source_run_dir"

    echo
    echo "Validating analysed run..."
    validate_run_dir "$source_run_dir"

    event_count="$(count_events "$source_run_dir")"
    echo "Event count: $event_count"

    if [[ "$event_count" -le 0 ]]; then
      echo "ERROR: event count is zero; refusing to accept run." >&2
      exit 1
    fi

    echo
    echo "Moving run to master output:"
    echo "  source: $source_run_dir"
    echo "  dest:   $dest_run_dir"

    if [[ -e "$dest_run_dir" ]]; then
      echo "ERROR: destination already exists: $dest_run_dir" >&2
      exit 1
    fi

    mv "$source_run_dir" "$dest_run_dir"

    echo
    echo "Run stored at: $dest_run_dir"

    echo
    echo "nvidia-smi after run:"
    nvidia-smi || true

    echo
    echo "Cleaning up servers after run..."
    kill_old_servers

    echo
    echo "Run $run_label completed successfully."

  } > "$run_log" 2>&1 && run_status="OK" || run_status="FAILED"

  end_epoch=$(date +%s)
  end_time="$(timestamp)"
  duration_s=$(( end_epoch - start_epoch ))

  if [[ "$run_status" == "OK" ]]; then
    success_count=$(( success_count + 1 ))

    if [[ -d "$dest_run_dir" ]]; then
      event_count="$(count_events "$dest_run_dir")"
    fi

    echo "[$end_time] $run_label completed successfully in ${duration_s}s. Events=$event_count"
  else
    fail_count=$(( fail_count + 1 ))
    echo "[$end_time] $run_label FAILED after ${duration_s}s. See log: $run_log" >&2

    failed_latest="$(latest_matrix_dir)"
    if [[ -n "$failed_latest" && -d "$failed_latest" ]]; then
      failed_dest="${MASTER_OUT}/${run_label}_FAILED_partial"
      if [[ ! -e "$failed_dest" ]]; then
        echo "Saving partial failed run: $failed_latest -> $failed_dest"
        mv "$failed_latest" "$failed_dest" 2>/dev/null || true
      fi
    fi

    kill_old_servers
  fi

  echo "${run_idx},${run_status},${source_run_dir},${dest_run_dir},${event_count},${start_time},${end_time},${duration_s}" >> "$STATUS_CSV"

  sleep 3
done

{
  echo
  echo "======================================================================"
  echo "Repeated experiment complete"
  echo "======================================================================"
  echo "End time:        $(date)"
  echo "Requested runs:  $N_RUNS"
  echo "Successful runs: $success_count"
  echo "Failed runs:     $fail_count"
  echo "Master output:   $MASTER_OUT"
  echo
  echo "Run status:"
  cat "$STATUS_CSV"
  echo
  echo "nvidia-smi after all runs:"
  nvidia-smi || true
} | tee "${MASTER_OUT}/final_summary.txt"

echo
echo "All done."
echo "Master output directory:"
echo "$MASTER_OUT"
