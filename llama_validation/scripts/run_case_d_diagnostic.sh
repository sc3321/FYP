#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/sc3321/FYP/llama_validation"
SERVER="${ROOT}/llama.cpp/build/bin/llama-server"
MODEL="${ROOT}/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${ROOT}/runs/diagnostic_caseD_${TIMESTAMP}"
POLICY_LOG="${OUT_ROOT}/policy_counters.log"

LC_PORT=8080
BE_PORT=8081

# Same parameters as the main 50-run matrix script so the regime matches.
N_LC=50
N_BE_LONG=25
N_WARMUP_LC=3
N_WARMUP_BE=2

SHM_NAME="/gpuphase_gpu0"
BE_DELAY_US=5000
MAX_DELAY_LOOPS=100000
BE_LONG_LIMIT=1
N_GPU_LAYERS=99

# Diagnostic-only: enable 1Hz policy counter sampling in both servers.
POLICY_SAMPLE_MS=1000

mkdir -p "$OUT_ROOT"

echo "================================================================"
echo "Diagnostic run: caseD with policy counter sampling"
echo "Output dir:    $OUT_ROOT"
echo "Policy log:    $POLICY_LOG"
echo "Sample rate:   ${POLICY_SAMPLE_MS}ms"
echo "================================================================"

if [[ ! -x "$SERVER" ]]; then
  echo "ERROR: llama-server not executable at $SERVER" >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: model not found at $MODEL" >&2
  exit 1
fi

LC_PID=""
BE_PID=""

cleanup_servers() {
  if [[ "${LC_PID:-}" != "" ]]; then
    kill "$LC_PID" 2>/dev/null || true
    wait "$LC_PID" 2>/dev/null || true
  fi

  if [[ "${BE_PID:-}" != "" ]]; then
    kill "$BE_PID" 2>/dev/null || true
    wait "$BE_PID" 2>/dev/null || true
  fi

  LC_PID=""
  BE_PID=""
}

kill_stale_servers() {
  cleanup_servers

  pkill -u "$USER" -f "llama-server" 2>/dev/null || true

  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${LC_PORT}/tcp" 2>/dev/null || true
    fuser -k "${BE_PORT}/tcp" 2>/dev/null || true
  fi

  sleep 1
}

reset_shared_memory() {
  kill_stale_servers
  rm -f /dev/shm/gpuphase_gpu0 2>/dev/null || true
}

trap cleanup_servers EXIT

wait_for_server() {
  local port="$1"
  local name="$2"
  local pid="$3"
  local log_dir="$4"

  echo "Waiting for $name server on port $port..."

  for i in $(seq 1 120); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: $name server died before becoming ready" >&2
      tail -120 "${log_dir}/server_stderr.log" >&2 || true
      exit 1
    fi

    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "$name ready."
      return 0
    fi

    sleep 1
  done

  echo "ERROR: $name did not become ready" >&2
  tail -120 "${log_dir}/server_stderr.log" >&2 || true
  exit 1
}

start_lc_server() {
  local log_dir="$1"
  mkdir -p "$log_dir"

  CUDA_VISIBLE_DEVICES=0 \
  GPU_PHASE_SHM_NAME="$SHM_NAME" \
  GPU_PHASE_LOG_DIR="$log_dir" \
  GPU_PHASE_POLICY_LOG="$POLICY_LOG" \
  GPU_PHASE_POLICY_SAMPLE_MS="$POLICY_SAMPLE_MS" \
  POLICY_MODE="CAP" \
  BE_DELAY_US="$BE_DELAY_US" \
  GPU_PHASE_BE_LONG_LIMIT="$BE_LONG_LIMIT" \
  GPU_PHASE_MAX_DELAY_LOOPS="$MAX_DELAY_LOOPS" \
  GPU_PHASE_WORKLOAD_CLASS=LC \
  GPU_PHASE_GRANULARITY=SHORT \
  "$SERVER" \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$LC_PORT" \
    -ngl "$N_GPU_LAYERS" \
    -np 1 \
    > "${log_dir}/server_stdout.log" \
    2> "${log_dir}/server_stderr.log" &

  LC_PID=$!
  wait_for_server "$LC_PORT" "LC" "$LC_PID" "$log_dir"
  echo "LC server pid: $LC_PID"
}

start_be_server() {
  local log_dir="$1"
  mkdir -p "$log_dir"

  CUDA_VISIBLE_DEVICES=0 \
  GPU_PHASE_SHM_NAME="$SHM_NAME" \
  GPU_PHASE_LOG_DIR="$log_dir" \
  GPU_PHASE_POLICY_LOG="$POLICY_LOG" \
  GPU_PHASE_POLICY_SAMPLE_MS="$POLICY_SAMPLE_MS" \
  POLICY_MODE="CAP" \
  BE_DELAY_US="$BE_DELAY_US" \
  GPU_PHASE_BE_LONG_LIMIT="$BE_LONG_LIMIT" \
  GPU_PHASE_MAX_DELAY_LOOPS="$MAX_DELAY_LOOPS" \
  GPU_PHASE_WORKLOAD_CLASS=BE \
  GPU_PHASE_GRANULARITY=LONG \
  "$SERVER" \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$BE_PORT" \
    -ngl "$N_GPU_LAYERS" \
    -np 1 \
    > "${log_dir}/server_stdout.log" \
    2> "${log_dir}/server_stderr.log" &

  BE_PID=$!
  wait_for_server "$BE_PORT" "BE" "$BE_PID" "$log_dir"
  echo "BE server pid: $BE_PID"
}

request_payload() {
  local kind="$1"

  if [[ "$kind" == "lc" ]]; then
    cat <<'JSON'
{"prompt": "Explain GPU kernels and GPU scheduling in two short paragraphs.", "n_predict": 128, "stream": false}
JSON

  elif [[ "$kind" == "be_long" ]]; then
    cat <<'JSON'
{"prompt": "Write a detailed technical explanation of transformer inference, including prefill, decode, batching, KV cache behaviour, GPU execution, memory bandwidth, scheduling interference, latency-critical work, best-effort work, and datacenter GPU sharing.", "n_predict": 512, "stream": false}
JSON

  elif [[ "$kind" == "warmup" ]]; then
    cat <<'JSON'
{"prompt": "Briefly mention one thing.", "n_predict": 16, "stream": false}
JSON

  else
    echo "ERROR: unknown request kind: $kind" >&2
    exit 1
  fi
}

warmup_client() {
  local port="$1"
  local n="$2"
  local label="$3"

  if [[ "$n" -le 0 ]]; then
    return 0
  fi

  echo "Warming up $label ($n requests)..."

  for i in $(seq 1 "$n"); do
    request_payload "warmup" | curl -fsS "http://127.0.0.1:${port}/completion" \
      -H "Content-Type: application/json" \
      -d @- \
      > /dev/null 2>&1 || true
  done
}

run_client() {
  local port="$1"
  local kind="$2"
  local n="$3"
  local out="$4"

  local out_dir
  out_dir="$(dirname "$out")"

  local resp_dir="${out_dir}/responses_${kind}"

  mkdir -p "$out_dir" "$resp_dir"
  : > "$out"

  for i in $(seq 1 "$n"); do
    local response_file="${resp_dir}/${kind}_${i}.json"

    local start_ns
    start_ns="$(date +%s%N)"

    request_payload "$kind" | curl -fsS "http://127.0.0.1:${port}/completion" \
      -H "Content-Type: application/json" \
      -d @- \
      > "$response_file" || true

    local end_ns
    end_ns="$(date +%s%N)"

    local latency_ms
    latency_ms=$(( (end_ns - start_ns) / 1000000 ))

    printf '{"i":%d,"kind":"%s","port":%d,"latency_ms":%d,"start_ns":%s,"end_ns":%s}\n' \
      "$i" "$kind" "$port" "$latency_ms" "$start_ns" "$end_ns" >> "$out"
  done
}

# ----- Run -----

reset_shared_memory

case_dir="${OUT_ROOT}/caseD_lc_be_long_policy"
mkdir -p "$case_dir"

# Mark when the experiment proper begins, for time series alignment.
echo "[$(date +%s.%N)] EXPERIMENT_START" >> "$POLICY_LOG"

start_lc_server "${case_dir}/lc_events"
start_be_server "${case_dir}/be_events"

echo "[$(date +%s.%N)] BOTH_SERVERS_READY lc_pid=$LC_PID be_pid=$BE_PID" >> "$POLICY_LOG"

warmup_client "$LC_PORT" "$N_WARMUP_LC" "LC"
warmup_client "$BE_PORT" "$N_WARMUP_BE" "BE"

echo "[$(date +%s.%N)] WARMUP_COMPLETE" >> "$POLICY_LOG"

run_client "$BE_PORT" "be_long" "$N_BE_LONG" "${case_dir}/be_client.jsonl" &
c_be=$!

sleep 0.2

run_client "$LC_PORT" "lc" "$N_LC" "${case_dir}/lc_client.jsonl" &
c_lc=$!

echo "[$(date +%s.%N)] CLIENTS_LAUNCHED" >> "$POLICY_LOG"

wait "$c_be"
wait "$c_lc"

echo "[$(date +%s.%N)] CLIENTS_COMPLETE" >> "$POLICY_LOG"

# Snapshots fire on server shutdown when cleanup() runs.
cleanup_servers

echo "[$(date +%s.%N)] EXPERIMENT_END" >> "$POLICY_LOG"

echo ""
echo "================================================================"
echo "Done. Inspect:"
echo "  Policy counter log: $POLICY_LOG"
echo "  Case output dir:    $case_dir"
echo "================================================================"
echo ""
echo "Quick summary:"
echo "  Total snapshots:    $(grep -c '^\[PolicyCounters\]' "$POLICY_LOG" 2>/dev/null || echo 0)"
echo "  LC pid snapshots:   $(grep -c "pid=$LC_PID" "$POLICY_LOG" 2>/dev/null || echo 0)"
echo "  BE pid snapshots:   $(grep -c "pid=$BE_PID" "$POLICY_LOG" 2>/dev/null || echo 0)"
