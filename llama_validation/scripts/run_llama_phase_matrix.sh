#!/usr/bin/env bash
set -euo pipefail

ROOT="/vol/bitbucket/sc3321/FYP/llama_validation"
SERVER="${ROOT}/llama.cpp/build/bin/llama-server"
MODEL="${ROOT}/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

OUT_ROOT="${ROOT}/runs/llama_phase_matrix_$(date +%Y%m%d_%H%M%S)"

LC_PORT=8080
BE_PORT=8081

N_LC=10
N_BE_LONG=5
N_BE_SHORT=20

N_LC_LONG=3
N_BE_LONG_TRIGGER=3

SHM_NAME="/gpuphase_gpu0"
BE_DELAY_US=5000
MAX_DELAY_LOOPS=100000
BE_LONG_LIMIT=1

# Change this to 20/40 if you hit CUDA OOM with two servers.
N_GPU_LAYERS=99

mkdir -p "$OUT_ROOT"

echo "Output directory: $OUT_ROOT"
echo "Server: $SERVER"
echo "Model: $MODEL"
echo "GPU layers: $N_GPU_LAYERS"

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

  # Kill any previous llama-server instances owned by this user.
  pkill -u "$USER" -f "llama-server" 2>/dev/null || true

  # Also clear port users if fuser exists.
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
      echo "ERROR: $name server process died before becoming ready. pid=$pid" >&2
      echo "===== ${name} stderr =====" >&2
      tail -120 "${log_dir}/server_stderr.log" >&2 || true
      echo "===== ${name} stdout =====" >&2
      tail -120 "${log_dir}/server_stdout.log" >&2 || true
      exit 1
    fi

    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "$name server ready."
      return 0
    fi

    if curl -fsS "http://127.0.0.1:${port}/props" >/dev/null 2>&1; then
      echo "$name server ready."
      return 0
    fi

    sleep 1
  done

  echo "ERROR: $name server did not become ready on port $port" >&2
  echo "===== ${name} stderr =====" >&2
  tail -120 "${log_dir}/server_stderr.log" >&2 || true
  echo "===== ${name} stdout =====" >&2
  tail -120 "${log_dir}/server_stdout.log" >&2 || true
  exit 1
}

start_lc_server() {
  local log_dir="$1"
  local policy="$2"

  mkdir -p "$log_dir"

  echo "Starting LC server: log_dir=$log_dir POLICY_MODE=$policy"

  GPU_PHASE_SHM_NAME="$SHM_NAME" \
  GPU_PHASE_LOG_DIR="$log_dir" \
  POLICY_MODE="$policy" \
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
}

start_be_server() {
  local log_dir="$1"
  local policy="$2"
  local gran="$3"

  mkdir -p "$log_dir"

  echo "Starting BE server: log_dir=$log_dir POLICY_MODE=$policy granularity=$gran"

  GPU_PHASE_SHM_NAME="$SHM_NAME" \
  GPU_PHASE_LOG_DIR="$log_dir" \
  POLICY_MODE="$policy" \
  BE_DELAY_US="$BE_DELAY_US" \
  GPU_PHASE_BE_LONG_LIMIT="$BE_LONG_LIMIT" \
  GPU_PHASE_MAX_DELAY_LOOPS="$MAX_DELAY_LOOPS" \
  GPU_PHASE_WORKLOAD_CLASS=BE \
  GPU_PHASE_GRANULARITY="$gran" \
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
}

request_payload() {
  local kind="$1"

  if [[ "$kind" == "lc" ]]; then
    cat <<'JSON'
{
  "prompt": "Explain GPU kernels and GPU scheduling in two short paragraphs.",
  "n_predict": 128,
  "stream": false
}
JSON

  elif [[ "$kind" == "lc_long" ]]; then
    cat <<'JSON'
{
  "prompt": "Explain GPU scheduling, GPU kernel execution, latency-critical inference, best-effort background work, and why operating systems struggle to reason about accelerator execution. Be technically detailed.",
  "n_predict": 512,
  "stream": false
}
JSON

  elif [[ "$kind" == "be_long" ]]; then
    cat <<'JSON'
{
  "prompt": "Write a detailed technical explanation of transformer inference, including prefill, decode, batching, KV cache behaviour, GPU execution, memory bandwidth, scheduling interference, latency-critical work, best-effort work, and datacenter GPU sharing.",
  "n_predict": 512,
  "stream": false
}
JSON

  elif [[ "$kind" == "be_short" ]]; then
    cat <<'JSON'
{
  "prompt": "Summarise one aspect of transformer inference.",
  "n_predict": 64,
  "stream": false
}
JSON

  else
    echo "unknown kind: $kind" >&2
    exit 1
  fi
}

validate_response() {
  local response_file="$1"
  local kind="$2"
  local port="$3"
  local i="$4"

  if [[ ! -s "$response_file" ]]; then
    echo "ERROR: empty response for kind=$kind port=$port request=$i" >&2
    return 1
  fi

  # Only fail on an actual JSON error field, not the word "error" inside generated text.
  if grep -qiE '"error"[[:space:]]*:' "$response_file"; then
    echo "ERROR: bad response for kind=$kind port=$port request=$i" >&2
    cat "$response_file" >&2
    return 1
  fi

  if ! grep -q '"content"' "$response_file"; then
    echo "WARNING: response for kind=$kind port=$port request=$i does not contain \"content\"" >&2
    head -c 500 "$response_file" >&2 || true
    echo >&2
  fi
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
    local start_ns
    local end_ns
    local latency_ms
    local response_file

    response_file="${resp_dir}/${kind}_${i}.json"

    start_ns=$(date +%s%N)

    if ! request_payload "$kind" | curl -fsS "http://127.0.0.1:${port}/completion" \
      -H "Content-Type: application/json" \
      -d @- > "$response_file"; then
      echo "ERROR: curl failed for kind=$kind port=$port request=$i" >&2
      exit 1
    fi

    end_ns=$(date +%s%N)
    latency_ms=$(( (end_ns - start_ns) / 1000000 ))

    validate_response "$response_file" "$kind" "$port" "$i"

    printf '{"i":%d,"kind":"%s","port":%d,"latency_ms":%d,"start_ns":%s,"end_ns":%s,"response_file":"%s"}\n' \
      "$i" "$kind" "$port" "$latency_ms" "$start_ns" "$end_ns" "$response_file" >> "$out"
  done
}

save_config() {
  local case_dir="$1"
  local case_name="$2"
  local policy="$3"
  local be_gran="${4:-none}"
  local ordering="${5:-default}"

  cat > "${case_dir}/config.txt" <<CONFIG
case=${case_name}
model=${MODEL}
server=${SERVER}
lc_port=${LC_PORT}
be_port=${BE_PORT}
policy=${policy}
be_granularity=${be_gran}
ordering=${ordering}
n_lc=${N_LC}
n_be_long=${N_BE_LONG}
n_be_short=${N_BE_SHORT}
n_lc_long=${N_LC_LONG}
n_be_long_trigger=${N_BE_LONG_TRIGGER}
shm_name=${SHM_NAME}
be_delay_us=${BE_DELAY_US}
max_delay_loops=${MAX_DELAY_LOOPS}
be_long_limit=${BE_LONG_LIMIT}
ngl=${N_GPU_LAYERS}
np=1
CONFIG
}

check_case_events() {
  local case_dir="$1"
  local expected_min="$2"

  local count
  count=$(grep -R "Event type =" "$case_dir" 2>/dev/null | wc -l || true)

  echo "Event count for $(basename "$case_dir"): $count"

  if [[ "$count" -lt "$expected_min" ]]; then
    echo "ERROR: too few phase events in $case_dir. Expected at least $expected_min, got $count" >&2
    echo "This usually means instrumentation was not hit, the wrong server handled the request, or responses were errors." >&2
    exit 1
  fi
}

run_case_lc_alone() {
  local case_dir="${OUT_ROOT}/caseA_lc_alone_none"
  mkdir -p "$case_dir"

  echo "===== CASE A: LC alone, no policy ====="
  reset_shared_memory
  save_config "$case_dir" "lc_alone_none" "NONE" "none" "single"

  start_lc_server "${case_dir}/lc_events" "NONE"
  run_client "$LC_PORT" "lc" "$N_LC" "${case_dir}/lc_client.jsonl"

  check_case_events "$case_dir" 20
  cleanup_servers
}

run_case_be_long_alone() {
  local case_dir="${OUT_ROOT}/caseB_be_long_alone_none"
  mkdir -p "$case_dir"

  echo "===== CASE B: BE-long alone, no policy ====="
  reset_shared_memory
  save_config "$case_dir" "be_long_alone_none" "NONE" "LONG" "single"

  start_be_server "${case_dir}/be_events" "NONE" "LONG"
  run_client "$BE_PORT" "be_long" "$N_BE_LONG" "${case_dir}/be_client.jsonl"

  check_case_events "$case_dir" 10
  cleanup_servers
}

run_case_lc_be_long_none_be_first() {
  local case_dir="${OUT_ROOT}/caseC_lc_be_long_none"
  mkdir -p "$case_dir"

  echo "===== CASE C: LC + BE-long, no policy, BE-first ====="
  reset_shared_memory
  save_config "$case_dir" "lc_be_long_none" "NONE" "LONG" "be_first"

  start_lc_server "${case_dir}/lc_events" "NONE"
  start_be_server "${case_dir}/be_events" "NONE" "LONG"

  run_client "$BE_PORT" "be_long" "$N_BE_LONG" "${case_dir}/be_client.jsonl" &
  local c_be=$!

  sleep 0.2

  run_client "$LC_PORT" "lc" "$N_LC" "${case_dir}/lc_client.jsonl" &
  local c_lc=$!

  wait "$c_be"
  wait "$c_lc"

  check_case_events "$case_dir" 30
  cleanup_servers
}

run_case_lc_be_long_policy_be_first() {
  local case_dir="${OUT_ROOT}/caseD_lc_be_long_policy"
  mkdir -p "$case_dir"

  echo "===== CASE D: LC + BE-long, CAP policy, BE-first ====="
  reset_shared_memory
  save_config "$case_dir" "lc_be_long_policy" "CAP" "LONG" "be_first"

  start_lc_server "${case_dir}/lc_events" "CAP"
  start_be_server "${case_dir}/be_events" "CAP" "LONG"

  run_client "$BE_PORT" "be_long" "$N_BE_LONG" "${case_dir}/be_client.jsonl" &
  local c_be=$!

  sleep 0.2

  run_client "$LC_PORT" "lc" "$N_LC" "${case_dir}/lc_client.jsonl" &
  local c_lc=$!

  wait "$c_be"
  wait "$c_lc"

  check_case_events "$case_dir" 30
  cleanup_servers
}

run_case_lc_be_short_none() {
  local case_dir="${OUT_ROOT}/caseE_lc_be_short_none"
  mkdir -p "$case_dir"

  echo "===== CASE E: LC + BE-short, no policy ====="
  reset_shared_memory
  save_config "$case_dir" "lc_be_short_none" "NONE" "SHORT" "be_first"

  start_lc_server "${case_dir}/lc_events" "NONE"
  start_be_server "${case_dir}/be_events" "NONE" "SHORT"

  run_client "$BE_PORT" "be_short" "$N_BE_SHORT" "${case_dir}/be_client.jsonl" &
  local c_be=$!

  sleep 0.2

  run_client "$LC_PORT" "lc" "$N_LC" "${case_dir}/lc_client.jsonl" &
  local c_lc=$!

  wait "$c_be"
  wait "$c_lc"

  check_case_events "$case_dir" 60
  cleanup_servers
}

run_case_lc_be_short_policy() {
  local case_dir="${OUT_ROOT}/caseF_lc_be_short_policy"
  mkdir -p "$case_dir"

  echo "===== CASE F: LC + BE-short, CAP policy ====="
  reset_shared_memory
  save_config "$case_dir" "lc_be_short_policy" "CAP" "SHORT" "be_first"

  start_lc_server "${case_dir}/lc_events" "CAP"
  start_be_server "${case_dir}/be_events" "CAP" "SHORT"

  run_client "$BE_PORT" "be_short" "$N_BE_SHORT" "${case_dir}/be_client.jsonl" &
  local c_be=$!

  sleep 0.2

  run_client "$LC_PORT" "lc" "$N_LC" "${case_dir}/lc_client.jsonl" &
  local c_lc=$!

  wait "$c_be"
  wait "$c_lc"

  check_case_events "$case_dir" 60
  cleanup_servers
}

run_case_lc_first_be_long_none() {
  local case_dir="${OUT_ROOT}/caseG_lc_first_be_long_none"
  mkdir -p "$case_dir"

  echo "===== CASE G: LC-first + BE-long, no policy ====="
  reset_shared_memory
  save_config "$case_dir" "lc_first_be_long_none" "NONE" "LONG" "lc_first"

  start_lc_server "${case_dir}/lc_events" "NONE"
  start_be_server "${case_dir}/be_events" "NONE" "LONG"

  run_client "$LC_PORT" "lc_long" "$N_LC_LONG" "${case_dir}/lc_client.jsonl" &
  local c_lc=$!

  sleep 0.05

  run_client "$BE_PORT" "be_long" "$N_BE_LONG_TRIGGER" "${case_dir}/be_client.jsonl" &
  local c_be=$!

  wait "$c_lc"
  wait "$c_be"

  check_case_events "$case_dir" 12
  cleanup_servers
}

run_case_lc_first_be_long_policy() {
  local case_dir="${OUT_ROOT}/caseH_lc_first_be_long_policy"
  mkdir -p "$case_dir"

  echo "===== CASE H: LC-first + BE-long, CAP policy ====="
  reset_shared_memory
  save_config "$case_dir" "lc_first_be_long_policy" "CAP" "LONG" "lc_first"

  start_lc_server "${case_dir}/lc_events" "CAP"
  start_be_server "${case_dir}/be_events" "CAP" "LONG"

  run_client "$LC_PORT" "lc_long" "$N_LC_LONG" "${case_dir}/lc_client.jsonl" &
  local c_lc=$!

  sleep 0.05

  run_client "$BE_PORT" "be_long" "$N_BE_LONG_TRIGGER" "${case_dir}/be_client.jsonl" &
  local c_be=$!

  wait "$c_lc"
  wait "$c_be"

  check_case_events "$case_dir" 12
  cleanup_servers
}

run_case_lc_alone
run_case_be_long_alone
run_case_lc_be_long_none_be_first
run_case_lc_be_long_policy_be_first
run_case_lc_be_short_none
run_case_lc_be_short_policy
run_case_lc_first_be_long_none
run_case_lc_first_be_long_policy

echo "All runs complete."
echo "Results in: $OUT_ROOT"
