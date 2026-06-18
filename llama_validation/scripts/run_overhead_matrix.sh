#!/usr/bin/env bash
# Abstraction overhead measurement: vanilla vs instrumented llama-server.
# Case A conditions only (LC alone, no BE peer, no contention).
# Interleaved replicates with alternating arm order to absorb time-of-run drift.

set -euo pipefail

# ---------- Paths -------------------------------------------------------------
ROOT="/home/sc3321/FYP/llama_validation"
INSTR_SERVER="${INSTR_SERVER:-${ROOT}/llama.cpp/build/bin/llama-server}"
VANILLA_SERVER="${VANILLA_SERVER:-${HOME}/llama-vanilla/build/bin/llama-server}"
MODEL="${MODEL:-${ROOT}/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"

# ---------- Knobs (match Case A) ----------------------------------------------
N_REPS="${N_REPS:-30}"
N_LC=50
N_WARMUP=3
LC_PORT=8080
N_GPU_LAYERS=99

# Instrumented arm only. Vanilla ignores these.
SHM_NAME="/sharedMemName"
POLICY_MODE_INSTR="${POLICY_MODE_INSTR:-CAP}"   # CAP = full bookkeeping path
BE_DELAY_US=5000
MAX_DELAY_LOOPS=100000
POLICY_SAMPLE_MS=1000

OUT_ROOT="${ROOT}/runs/overhead_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"

# ---------- Sanity checks -----------------------------------------------------
for b in "$VANILLA_SERVER" "$INSTR_SERVER"; do
  if [[ ! -x "$b" ]]; then
    echo "ERROR: server binary missing or not executable: $b" >&2
    exit 1
  fi
done
if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: model not found at $MODEL" >&2
  exit 1
fi

# ---------- Metadata ----------------------------------------------------------
git_head_or_unknown() {
  local p="$1"
  if [[ -d "$p/.git" ]]; then
    git -C "$p" rev-parse HEAD 2>/dev/null || echo unknown
  else
    echo unknown
  fi
}

VANILLA_REPO="$(dirname "$VANILLA_SERVER")/../.."
INSTR_REPO="$(dirname "$INSTR_SERVER")/../.."

{
  echo "experiment=abstraction_overhead"
  echo "host=$(hostname)"
  echo "user=${USER}"
  echo "date=$(date -Is)"
  echo "n_reps=${N_REPS}"
  echo "n_lc=${N_LC}"
  echo "n_warmup=${N_WARMUP}"
  echo "policy_mode_instr=${POLICY_MODE_INSTR}"
  echo "ngl=${N_GPU_LAYERS}"
  echo "vanilla_server=${VANILLA_SERVER}"
  echo "vanilla_md5=$(md5sum "$VANILLA_SERVER" | awk '{print $1}')"
  echo "vanilla_size_bytes=$(stat -c%s "$VANILLA_SERVER")"
  echo "vanilla_commit=$(git_head_or_unknown "$VANILLA_REPO")"
  echo "instr_server=${INSTR_SERVER}"
  echo "instr_md5=$(md5sum "$INSTR_SERVER" | awk '{print $1}')"
  echo "instr_size_bytes=$(stat -c%s "$INSTR_SERVER")"
  echo "instr_commit=$(git_head_or_unknown "$INSTR_REPO")"
  echo "model=${MODEL}"
  echo "model_md5=$(md5sum "$MODEL" | awk '{print $1}')"
} > "$OUT_ROOT/metadata.txt"

echo "================================================================="
echo "Overhead experiment"
echo "================================================================="
cat "$OUT_ROOT/metadata.txt"
echo "Output: $OUT_ROOT"
echo "================================================================="

# ---------- Server lifecycle --------------------------------------------------
SERVER_PID=""

cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

kill_stale() {
  cleanup_server
  pkill -u "$USER" -f "llama-server" 2>/dev/null || true
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${LC_PORT}/tcp" 2>/dev/null || true
  fi
  sleep 1
}

reset_state() {
  kill_stale
  rm -f /dev/shm/gpuphase_gpu0 2>/dev/null || true
  rm -f /dev/shm/sharedMemName 2>/dev/null || true
}

trap 'cleanup_server' EXIT

wait_for_server() {
  local pid="$1"
  local log_dir="$2"
  for i in $(seq 1 120); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: server died before becoming ready" >&2
      tail -120 "$log_dir/server_stderr.log" >&2 || true
      exit 1
    fi
    if curl -fsS "http://127.0.0.1:${LC_PORT}/health" >/dev/null 2>&1 \
    || curl -fsS "http://127.0.0.1:${LC_PORT}/props"  >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "ERROR: server not ready after 120s" >&2
  tail -120 "$log_dir/server_stderr.log" >&2 || true
  exit 1
}

start_vanilla() {
  local log_dir="$1"
  mkdir -p "$log_dir"
  CUDA_VISIBLE_DEVICES=0 \
  "$VANILLA_SERVER" \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$LC_PORT" \
    -ngl "$N_GPU_LAYERS" \
    -np 1 \
    > "$log_dir/server_stdout.log" \
    2> "$log_dir/server_stderr.log" &
  SERVER_PID=$!
  wait_for_server "$SERVER_PID" "$log_dir"
}

start_instr() {
  local log_dir="$1"
  local arm_dir="$2"
  mkdir -p "$log_dir"
  local policy_log="${arm_dir}/policy_counters.log"

  CUDA_VISIBLE_DEVICES=0 \
  GPU_PHASE_SHM_NAME="$SHM_NAME" \
  GPU_PHASE_LOG_DIR="$log_dir" \
  GPU_PHASE_POLICY_LOG="$policy_log" \
  GPU_PHASE_POLICY_SAMPLE_MS="$POLICY_SAMPLE_MS" \
  POLICY_MODE="$POLICY_MODE_INSTR" \
  BE_DELAY_US="$BE_DELAY_US" \
  GPU_PHASE_MAX_DELAY_LOOPS="$MAX_DELAY_LOOPS" \
  GPU_PHASE_WORKLOAD_CLASS=LC \
  GPU_PHASE_GRANULARITY=SHORT \
  "$INSTR_SERVER" \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$LC_PORT" \
    -ngl "$N_GPU_LAYERS" \
    -np 1 \
    > "$log_dir/server_stdout.log" \
    2> "$log_dir/server_stderr.log" &
  SERVER_PID=$!
  wait_for_server "$SERVER_PID" "$log_dir"
}

# ---------- Payloads (identical to your Case A) -------------------------------
lc_payload() {
  cat <<'JSON'
{ "prompt": "Explain GPU kernels and GPU scheduling in two short paragraphs.", "n_predict": 128, "stream": false }
JSON
}

warmup_payload() {
  cat <<'JSON'
{ "prompt": "Briefly mention one thing.", "n_predict": 16, "stream": false }
JSON
}

# ---------- Client ------------------------------------------------------------
validate_response() {
  local f="$1"; local i="$2"
  if [[ ! -s "$f" ]]; then
    echo "ERROR: empty response idx=$i" >&2
    return 1
  fi
  if grep -qiE '"error"[[:space:]]*:' "$f"; then
    echo "ERROR: error response idx=$i" >&2
    cat "$f" >&2
    return 1
  fi
  if ! grep -q '"content"' "$f"; then
    echo "WARNING: response idx=$i lacks content" >&2
  fi
}

warmup_client() {
  for i in $(seq 1 "$N_WARMUP"); do
    warmup_payload | curl -fsS "http://127.0.0.1:${LC_PORT}/completion" \
      -H "Content-Type: application/json" -d @- > /dev/null 2>&1 \
      || echo "WARNING: warmup $i failed" >&2
  done
}

run_lc_client() {
  local out="$1"
  local resp_dir="$2"
  mkdir -p "$resp_dir"
  : > "$out"
  for i in $(seq 1 "$N_LC"); do
    local rf="${resp_dir}/lc_${i}.json"
    local s e lat
    s=$(date +%s%N)
    if ! lc_payload | curl -fsS "http://127.0.0.1:${LC_PORT}/completion" \
      -H "Content-Type: application/json" -d @- > "$rf"; then
      echo "ERROR: curl failed idx=$i" >&2
      exit 1
    fi
    e=$(date +%s%N)
    lat=$(( (e - s) / 1000000 ))
    validate_response "$rf" "$i"
    printf '{"i":%d,"kind":"lc","port":%d,"latency_ms":%d,"start_ns":%s,"end_ns":%s,"response_file":"%s"}\n' \
      "$i" "$LC_PORT" "$lat" "$s" "$e" "$rf" >> "$out"
  done
}

# ---------- One arm -----------------------------------------------------------
run_arm() {
  local arm="$1"       # vanilla | instr
  local rep_dir="$2"
  local arm_dir="${rep_dir}/${arm}"
  local events_dir="${arm_dir}/lc_events"
  local resp_dir="${arm_dir}/responses_lc"
  mkdir -p "$arm_dir"

  cat > "${arm_dir}/config.txt" <<CFG
arm=${arm}
n_lc=${N_LC}
n_warmup=${N_WARMUP}
policy_mode=$([[ "$arm" == "instr" ]] && echo "$POLICY_MODE_INSTR" || echo "n/a")
binary=$([[ "$arm" == "instr" ]] && echo "$INSTR_SERVER" || echo "$VANILLA_SERVER")
CFG

  reset_state

  if [[ "$arm" == "vanilla" ]]; then
    start_vanilla "$events_dir"
  else
    start_instr "$events_dir" "$arm_dir"
  fi

  warmup_client
  run_lc_client "${arm_dir}/lc_client.jsonl" "$resp_dir"
  cleanup_server
}

# ---------- Main loop ---------------------------------------------------------
for rep in $(seq 1 "$N_REPS"); do
  rep_dir="${OUT_ROOT}/$(printf 'rep%02d' "$rep")"
  mkdir -p "$rep_dir"

  # Alternate first-arm each rep to cancel position bias.
  if (( rep % 2 == 1 )); then
    order=("vanilla" "instr")
  else
    order=("instr" "vanilla")
  fi

  echo
  echo "###################################################################"
  printf "# REP %02d / %02d   order: %s -> %s\n" "$rep" "$N_REPS" "${order[0]}" "${order[1]}"
  echo "###################################################################"

  for arm in "${order[@]}"; do
    echo "--- $arm ---"
    run_arm "$arm" "$rep_dir"
  done
done

echo
echo "All replicates complete."
echo "Output: $OUT_ROOT"
