#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/sc3321/FYP/llama_validation"
SERVER="${ROOT}/llama.cpp/build/bin/llama-server"
MODEL="${ROOT}/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Base output dir. When N_REPS > 1 each replicate goes into a repNN/ subdir.
OUT_ROOT_BASE="${ROOT}/runs/llama_phase_matrix_$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="$OUT_ROOT_BASE"   # overridden per-rep below

LC_PORT=8080
BE_PORT=8081

# Usage:
#   ./script.sh all          # old matrix + new continuous-LC cases
#   ./script.sh base         # old matrix only
#   ./script.sh continuous   # new continuous-LC cases only
#
# With replicates / case filtering (preferred for the LC-side eBPF analysis):
#   N_REPS=10 CASES="C D E F J K" ./script.sh
#
#   N_REPS  - integer, how many independent matrix runs to perform back-to-back.
#             Each rep lives in OUT_ROOT_BASE/repNN/. Default 1 (no rep subdir).
#   CASES   - space-separated case letters. Overrides RUN_SET when set.
#             Letters map to the existing run_case_* functions; see
#             run_case_by_letter() at the bottom of this file.
RUN_SET="${1:-all}"
N_REPS="${N_REPS:-1}"
CASES="${CASES:-}"

# Increased request counts for statistically meaningful tail percentiles.
N_LC=50
N_BE_LONG=25
N_BE_SHORT=50

N_LC_LONG=30
N_BE_LONG_TRIGGER=15

# Warmup: throwaway requests issued before timed measurement begins.
N_WARMUP_LC=3
N_WARMUP_BE=2

SHM_NAME="/sharedMemName"
BE_DELAY_US=5000
MAX_DELAY_LOOPS=100000

# Diagnostic policy counter sampling. This gives each case its own
# policy_counters.log, useful for checking activeLC / BE admission behaviour.
POLICY_SAMPLE_MS=1000

# Change this to 20/40 if you hit CUDA OOM with two servers.
N_GPU_LAYERS=99

# New continuous-LC experiment controls.
LC_CONCURRENCY=4
LC_SERVER_PARALLEL=4
BE_SERVER_PARALLEL=1
LC_BE_LAUNCH_GAP_SEC=0.20

# ------------------------------------------------------------------------------
# eBPF tracing configuration
# ------------------------------------------------------------------------------
EBPF_TRACE="${EBPF_TRACE:-1}"
EBPF_TRACE_IOCTL="${EBPF_TRACE_IOCTL:-0}"
EBPF_TRACER="${EBPF_TRACER:-${ROOT}/../ebpf/syscallTrace/syscall_trace}"
EBPF_ATTACH_SETTLE_SEC="${EBPF_ATTACH_SETTLE_SEC:-0.5}"
EBPF_PID=""

mkdir -p "$OUT_ROOT_BASE"

echo "Output base directory: $OUT_ROOT_BASE"
echo "Server:                $SERVER"
echo "Model:                 $MODEL"
echo "Run set:               $RUN_SET"
echo "N_REPS:                $N_REPS"
echo "CASES (filter):        ${CASES:-<unset; using RUN_SET>}"
echo "GPU layers:            $N_GPU_LAYERS"
echo "N_LC:                  $N_LC (warmup: $N_WARMUP_LC)"
echo "N_BE_LONG:             $N_BE_LONG (warmup: $N_WARMUP_BE)"
echo "N_BE_SHORT:            $N_BE_SHORT (warmup: $N_WARMUP_BE)"
echo "N_LC_LONG:             $N_LC_LONG"
echo "N_BE_LONG_TRIGGER:     $N_BE_LONG_TRIGGER"
echo "BE_DELAY_US:           $BE_DELAY_US"
echo "POLICY_SAMPLE_MS:      $POLICY_SAMPLE_MS"
echo "LC_CONCURRENCY:        $LC_CONCURRENCY"
echo "LC_SERVER_PARALLEL:    $LC_SERVER_PARALLEL"
echo "BE_SERVER_PARALLEL:    $BE_SERVER_PARALLEL"
echo "LC_BE_LAUNCH_GAP_SEC:  $LC_BE_LAUNCH_GAP_SEC"
echo "EBPF_TRACE:            $EBPF_TRACE"
echo "EBPF_TRACE_IOCTL:      $EBPF_TRACE_IOCTL"
echo "EBPF_TRACER:           $EBPF_TRACER"

if [[ ! -x "$SERVER" ]]; then
  echo "ERROR: llama-server not executable at $SERVER" >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: model not found at $MODEL" >&2
  exit 1
fi

if [[ "$EBPF_TRACE" == "1" ]]; then
  if [[ ! -x "$EBPF_TRACER" ]]; then
    echo "ERROR: EBPF_TRACE=1 but tracer not executable at $EBPF_TRACER" >&2
    echo "Build it, or set EBPF_TRACE=0 to run without tracing." >&2
    exit 1
  fi
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
  rm -f /dev/shm/sharedMemName 2>/dev/null || true
}

# ------------------------------------------------------------------------------
# eBPF tracer attach / detach (unchanged)
# ------------------------------------------------------------------------------
start_ebpf_trace() {
  local case_dir="$1"

  [[ "$EBPF_TRACE" != "1" ]] && return 0

  local pids=()
  [[ -n "${BE_PID:-}" ]] && pids+=("$BE_PID")
  [[ -n "${LC_PID:-}" ]] && pids+=("$LC_PID")

  if [[ "${#pids[@]}" -eq 0 ]]; then
    echo "WARNING: start_ebpf_trace called with no server PIDs; skipping." >&2
    return 0
  fi

  echo "Attaching eBPF tracer to PIDs: ${pids[*]} (ioctl=$EBPF_TRACE_IOCTL)"
  
  printf "LC_PID=%s\n" "${LC_PID:-}" > "${case_dir}/pids.txt"
  printf "BE_PID=%s\n" "${BE_PID:-}" >> "${case_dir}/pids.txt"

  sudo -n env EBPF_WATCH_IOCTL="$EBPF_TRACE_IOCTL" \
    "$EBPF_TRACER" "${pids[@]}" \
    > "${case_dir}/ebpf_events.jsonl" \
    2> "${case_dir}/ebpf_stderr.log" &
  EBPF_PID=$!
  
  sleep "$EBPF_ATTACH_SETTLE_SEC"
  if ! kill -0 "$EBPF_PID" 2>/dev/null; then
    echo "ERROR: eBPF tracer died immediately after launch. stderr:" >&2
    tail -40 "${case_dir}/ebpf_stderr.log" >&2 || true
    EBPF_PID=""
    echo "WARNING: continuing this case WITHOUT eBPF tracing." >&2
    return 0
  fi
}

stop_ebpf_trace() {
  [[ -z "${EBPF_PID:-}" ]] && return 0
  kill -INT "$EBPF_PID" 2>/dev/null || true
  wait "$EBPF_PID" 2>/dev/null || true
  EBPF_PID=""
}

trap 'stop_ebpf_trace; cleanup_servers' EXIT

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
  local np="${3:-1}"

  mkdir -p "$log_dir"

  local case_dir
  case_dir="$(dirname "$log_dir")"
  local policy_log="${case_dir}/policy_counters.log"

  echo "Starting LC server: log_dir=$log_dir POLICY_MODE=$policy np=$np"

  CUDA_VISIBLE_DEVICES=0 \
  GPU_PHASE_SHM_NAME="$SHM_NAME" \
  GPU_PHASE_LOG_DIR="$log_dir" \
  GPU_PHASE_POLICY_LOG="$policy_log" \
  GPU_PHASE_POLICY_SAMPLE_MS="$POLICY_SAMPLE_MS" \
  POLICY_MODE="$policy" \
  BE_DELAY_US="$BE_DELAY_US" \
  GPU_PHASE_MAX_DELAY_LOOPS="$MAX_DELAY_LOOPS" \
  GPU_PHASE_WORKLOAD_CLASS=LC \
  GPU_PHASE_GRANULARITY=SHORT \
  "$SERVER" \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$LC_PORT" \
    -ngl "$N_GPU_LAYERS" \
    -np "$np" \
    > "${log_dir}/server_stdout.log" \
    2> "${log_dir}/server_stderr.log" &

  LC_PID=$!
  wait_for_server "$LC_PORT" "LC" "$LC_PID" "$log_dir"
}

start_be_server() {
  local log_dir="$1"
  local policy="$2"
  local gran="$3"
  local np="${4:-1}"

  mkdir -p "$log_dir"

  local case_dir
  case_dir="$(dirname "$log_dir")"
  local policy_log="${case_dir}/policy_counters.log"

  echo "Starting BE server: log_dir=$log_dir POLICY_MODE=$policy granularity=$gran np=$np"

  CUDA_VISIBLE_DEVICES=0 \
  GPU_PHASE_SHM_NAME="$SHM_NAME" \
  GPU_PHASE_LOG_DIR="$log_dir" \
  GPU_PHASE_POLICY_LOG="$policy_log" \
  GPU_PHASE_POLICY_SAMPLE_MS="$POLICY_SAMPLE_MS" \
  POLICY_MODE="$policy" \
  BE_DELAY_US="$BE_DELAY_US" \
  GPU_PHASE_MAX_DELAY_LOOPS="$MAX_DELAY_LOOPS" \
  GPU_PHASE_WORKLOAD_CLASS=BE \
  GPU_PHASE_GRANULARITY="$gran" \
  "$SERVER" \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$BE_PORT" \
    -ngl "$N_GPU_LAYERS" \
    -np "$np" \
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

  elif [[ "$kind" == "warmup" ]]; then
    cat <<'JSON'
{
  "prompt": "Briefly mention one thing.",
  "n_predict": 16,
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

warmup_client() {
  local port="$1"
  local n="$2"
  local label="$3"

  if [[ "$n" -le 0 ]]; then
    return 0
  fi

  echo "Warming up $label on port $port ($n requests)..."

  for i in $(seq 1 "$n"); do
    if ! request_payload "warmup" | curl -fsS "http://127.0.0.1:${port}/completion" \
      -H "Content-Type: application/json" \
      -d @- > /dev/null 2>&1; then
      echo "WARNING: warmup request $i failed for $label port=$port" >&2
    fi
  done

  echo "Warmup complete for $label."
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

run_client_concurrent() {
  local port="$1"
  local kind="$2"
  local n="$3"
  local out="$4"
  local concurrency="$5"

  if [[ "$concurrency" -le 1 ]]; then
    run_client "$port" "$kind" "$n" "$out"
    return 0
  fi

  local out_dir
  out_dir="$(dirname "$out")"

  local resp_dir="${out_dir}/responses_${kind}"
  local tmp_dir="${out_dir}/tmp_${kind}_jsonl"

  mkdir -p "$out_dir" "$resp_dir"
  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir"
  : > "$out"

  echo "Running concurrent client: kind=$kind port=$port n=$n concurrency=$concurrency"

  local running=0
  local failed=0

  for i in $(seq 1 "$n"); do
    (
      local start_ns
      local end_ns
      local latency_ms
      local response_file
      local line_file

      response_file="${resp_dir}/${kind}_${i}.json"
      line_file="${tmp_dir}/$(printf "%06d" "$i").jsonl"

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

      printf '{"i":%d,"kind":"%s","port":%d,"latency_ms":%d,"start_ns":%s,"end_ns":%s,"response_file":"%s","client_concurrency":%d}\n' \
        "$i" "$kind" "$port" "$latency_ms" "$start_ns" "$end_ns" "$response_file" "$concurrency" > "$line_file"
    ) &

    running=$((running + 1))

    if [[ "$running" -ge "$concurrency" ]]; then
      if ! wait -n; then
        failed=1
      fi
      running=$((running - 1))
    fi
  done

  while [[ "$running" -gt 0 ]]; do
    if ! wait -n; then
      failed=1
    fi
    running=$((running - 1))
  done

  if [[ "$failed" -ne 0 ]]; then
    echo "ERROR: one or more concurrent client requests failed for kind=$kind port=$port" >&2
    exit 1
  fi

  find "$tmp_dir" -type f -name '*.jsonl' | sort | xargs cat > "$out"
  rm -rf "$tmp_dir"
}

save_config() {
  local case_dir="$1"
  local case_name="$2"
  local policy="$3"
  local be_gran="${4:-none}"
  local ordering="${5:-default}"
  local lc_np="${6:-1}"
  local be_np="${7:-1}"
  local lc_conc="${8:-1}"
  local be_conc="${9:-1}"

  cat > "${case_dir}/config.txt" <<CONFIG
case=${case_name}
model=${MODEL}
server=${SERVER}
run_set=${RUN_SET}
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
n_warmup_lc=${N_WARMUP_LC}
n_warmup_be=${N_WARMUP_BE}
shm_name=${SHM_NAME}
be_delay_us=${BE_DELAY_US}
max_delay_loops=${MAX_DELAY_LOOPS}
policy_sample_ms=${POLICY_SAMPLE_MS}
ngl=${N_GPU_LAYERS}
lc_server_np=${lc_np}
be_server_np=${be_np}
lc_client_concurrency=${lc_conc}
be_client_concurrency=${be_conc}
ebpf_trace=${EBPF_TRACE}
ebpf_trace_ioctl=${EBPF_TRACE_IOCTL}
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

ebpf_quick_summary() {
  local case_dir="$1"
  local jf="${case_dir}/ebpf_events.jsonl"

  [[ "$EBPF_TRACE" != "1" ]] && return 0
  if [[ ! -s "$jf" ]]; then
    echo "eBPF: no events file (or empty) for $(basename "$case_dir")"
    return 0
  fi

  local total sleeps band
  total=$(wc -l < "$jf" 2>/dev/null || echo 0)
  sleeps=$(grep -c '"kind":"\(clock_\)\?nanosleep"' "$jf" 2>/dev/null || echo 0)
  band=$(grep '"kind":"\(clock_\)\?nanosleep"' "$jf" 2>/dev/null \
         | grep -oE '"dur_ns":[0-9]+' \
         | awk -F: '$2>=4000000 && $2<=8000000{c++} END{print c+0}')
  echo "eBPF: $(basename "$case_dir"): total_events=$total nanosleep_events=$sleeps policy_band_4-8ms=$band"
}

# ------------------------------------------------------------------------------
# Case functions (unchanged from original)
# ------------------------------------------------------------------------------

run_case_lc_alone() {
  local case_dir="${OUT_ROOT}/caseA_lc_alone_none"
  mkdir -p "$case_dir"

  echo "===== CASE A: LC alone, no policy ====="
  reset_shared_memory
  save_config "$case_dir" "lc_alone_none" "NONE" "none" "single" 1 1 1 1

  start_lc_server "${case_dir}/lc_events" "NONE" 1
  start_ebpf_trace "$case_dir"
  warmup_client "$LC_PORT" "$N_WARMUP_LC" "LC"
  run_client "$LC_PORT" "lc" "$N_LC" "${case_dir}/lc_client.jsonl"

  stop_ebpf_trace
  check_case_events "$case_dir" "$(( 2 * N_LC ))"
  ebpf_quick_summary "$case_dir"
  cleanup_servers
}

run_case_be_long_alone() {
  local case_dir="${OUT_ROOT}/caseB_be_long_alone_none"
  mkdir -p "$case_dir"

  echo "===== CASE B: BE-long alone, no policy ====="
  reset_shared_memory
  save_config "$case_dir" "be_long_alone_none" "NONE" "LONG" "single" 1 1 1 1

  start_be_server "${case_dir}/be_events" "NONE" "LONG" 1
  start_ebpf_trace "$case_dir"
  warmup_client "$BE_PORT" "$N_WARMUP_BE" "BE"
  run_client "$BE_PORT" "be_long" "$N_BE_LONG" "${case_dir}/be_client.jsonl"

  stop_ebpf_trace
  check_case_events "$case_dir" "$(( 2 * N_BE_LONG ))"
  ebpf_quick_summary "$case_dir"
  cleanup_servers
}

run_case_lc_be_long_none_be_first() {
  local case_dir="${OUT_ROOT}/caseC_lc_be_long_none"
  mkdir -p "$case_dir"

  echo "===== CASE C: LC + BE-long, no policy, BE-first ====="
  reset_shared_memory
  save_config "$case_dir" "lc_be_long_none" "NONE" "LONG" "be_first" 1 1 1 1

  start_lc_server "${case_dir}/lc_events" "NONE" 1
  start_be_server "${case_dir}/be_events" "NONE" "LONG" 1
  start_ebpf_trace "$case_dir"

  warmup_client "$LC_PORT" "$N_WARMUP_LC" "LC"
  warmup_client "$BE_PORT" "$N_WARMUP_BE" "BE"

  run_client "$BE_PORT" "be_long" "$N_BE_LONG" "${case_dir}/be_client.jsonl" &
  local c_be=$!

  sleep 0.2

  run_client "$LC_PORT" "lc" "$N_LC" "${case_dir}/lc_client.jsonl" &
  local c_lc=$!

  wait "$c_be"
  wait "$c_lc"

  stop_ebpf_trace
  check_case_events "$case_dir" "$(( 2 * (N_LC + N_BE_LONG) ))"
  ebpf_quick_summary "$case_dir"
  cleanup_servers
}

run_case_lc_be_long_policy_be_first() {
  local case_dir="${OUT_ROOT}/caseD_lc_be_long_policy"
  mkdir -p "$case_dir"

  echo "===== CASE D: LC + BE-long, CAP policy, BE-first ====="
  reset_shared_memory
  save_config "$case_dir" "lc_be_long_policy" "CAP" "LONG" "be_first" 1 1 1 1

  start_lc_server "${case_dir}/lc_events" "CAP" 1
  start_be_server "${case_dir}/be_events" "CAP" "LONG" 1
  start_ebpf_trace "$case_dir"

  warmup_client "$LC_PORT" "$N_WARMUP_LC" "LC"
  warmup_client "$BE_PORT" "$N_WARMUP_BE" "BE"

  run_client "$BE_PORT" "be_long" "$N_BE_LONG" "${case_dir}/be_client.jsonl" &
  local c_be=$!

  sleep 0.2

  run_client "$LC_PORT" "lc" "$N_LC" "${case_dir}/lc_client.jsonl" &
  local c_lc=$!

  wait "$c_be"
  wait "$c_lc"

  stop_ebpf_trace
  check_case_events "$case_dir" "$(( 2 * (N_LC + N_BE_LONG) ))"
  ebpf_quick_summary "$case_dir"
  cleanup_servers
}

run_case_lc_be_short_none() {
  local case_dir="${OUT_ROOT}/caseE_lc_be_short_none"
  mkdir -p "$case_dir"

  echo "===== CASE E: LC + BE-short, no policy ====="
  reset_shared_memory
  save_config "$case_dir" "lc_be_short_none" "NONE" "SHORT" "be_first" 1 1 1 1

  start_lc_server "${case_dir}/lc_events" "NONE" 1
  start_be_server "${case_dir}/be_events" "NONE" "SHORT" 1
  start_ebpf_trace "$case_dir"

  warmup_client "$LC_PORT" "$N_WARMUP_LC" "LC"
  warmup_client "$BE_PORT" "$N_WARMUP_BE" "BE"

  run_client "$BE_PORT" "be_short" "$N_BE_SHORT" "${case_dir}/be_client.jsonl" &
  local c_be=$!

  sleep 0.2

  run_client "$LC_PORT" "lc" "$N_LC" "${case_dir}/lc_client.jsonl" &
  local c_lc=$!

  wait "$c_be"
  wait "$c_lc"

  stop_ebpf_trace
  check_case_events "$case_dir" "$(( 2 * (N_LC + N_BE_SHORT) ))"
  ebpf_quick_summary "$case_dir"
  cleanup_servers
}

run_case_lc_be_short_policy() {
  local case_dir="${OUT_ROOT}/caseF_lc_be_short_policy"
  mkdir -p "$case_dir"

  echo "===== CASE F: LC + BE-short, CAP policy ====="
  reset_shared_memory
  save_config "$case_dir" "lc_be_short_policy" "CAP" "SHORT" "be_first" 1 1 1 1

  start_lc_server "${case_dir}/lc_events" "CAP" 1
  start_be_server "${case_dir}/be_events" "CAP" "SHORT" 1
  start_ebpf_trace "$case_dir"

  warmup_client "$LC_PORT" "$N_WARMUP_LC" "LC"
  warmup_client "$BE_PORT" "$N_WARMUP_BE" "BE"

  run_client "$BE_PORT" "be_short" "$N_BE_SHORT" "${case_dir}/be_client.jsonl" &
  local c_be=$!

  sleep 0.2

  run_client "$LC_PORT" "lc" "$N_LC" "${case_dir}/lc_client.jsonl" &
  local c_lc=$!

  wait "$c_be"
  wait "$c_lc"

  stop_ebpf_trace
  check_case_events "$case_dir" "$(( 2 * (N_LC + N_BE_SHORT) ))"
  ebpf_quick_summary "$case_dir"
  cleanup_servers
}

run_case_lc_first_be_long_none() {
  local case_dir="${OUT_ROOT}/caseG_lc_first_be_long_none"
  mkdir -p "$case_dir"

  echo "===== CASE G: LC-first + BE-long, no policy ====="
  reset_shared_memory
  save_config "$case_dir" "lc_first_be_long_none" "NONE" "LONG" "lc_first" 1 1 1 1

  start_lc_server "${case_dir}/lc_events" "NONE" 1
  start_be_server "${case_dir}/be_events" "NONE" "LONG" 1
  start_ebpf_trace "$case_dir"

  warmup_client "$LC_PORT" "$N_WARMUP_LC" "LC"
  warmup_client "$BE_PORT" "$N_WARMUP_BE" "BE"

  run_client "$LC_PORT" "lc_long" "$N_LC_LONG" "${case_dir}/lc_client.jsonl" &
  local c_lc=$!

  sleep 0.05

  run_client "$BE_PORT" "be_long" "$N_BE_LONG_TRIGGER" "${case_dir}/be_client.jsonl" &
  local c_be=$!

  wait "$c_lc"
  wait "$c_be"

  stop_ebpf_trace
  check_case_events "$case_dir" "$(( 2 * (N_LC_LONG + N_BE_LONG_TRIGGER) ))"
  ebpf_quick_summary "$case_dir"
  cleanup_servers
}

run_case_lc_first_be_long_policy() {
  local case_dir="${OUT_ROOT}/caseH_lc_first_be_long_policy"
  mkdir -p "$case_dir"

  echo "===== CASE H: LC-first + BE-long, CAP policy ====="
  reset_shared_memory
  save_config "$case_dir" "lc_first_be_long_policy" "CAP" "LONG" "lc_first" 1 1 1 1

  start_lc_server "${case_dir}/lc_events" "CAP" 1
  start_be_server "${case_dir}/be_events" "CAP" "LONG" 1
  start_ebpf_trace "$case_dir"

  warmup_client "$LC_PORT" "$N_WARMUP_LC" "LC"
  warmup_client "$BE_PORT" "$N_WARMUP_BE" "BE"

  run_client "$LC_PORT" "lc_long" "$N_LC_LONG" "${case_dir}/lc_client.jsonl" &
  local c_lc=$!

  sleep 0.05

  run_client "$BE_PORT" "be_long" "$N_BE_LONG_TRIGGER" "${case_dir}/be_client.jsonl" &
  local c_be=$!

  wait "$c_lc"
  wait "$c_be"

  stop_ebpf_trace
  check_case_events "$case_dir" "$(( 2 * (N_LC_LONG + N_BE_LONG_TRIGGER) ))"
  ebpf_quick_summary "$case_dir"
  cleanup_servers
}

run_case_lc_cont_alone_none() {
  local case_dir="${OUT_ROOT}/caseI_lc_cont_alone_none"
  mkdir -p "$case_dir"

  echo "===== CASE I: continuous LC alone, no policy ====="
  reset_shared_memory
  save_config "$case_dir" "lc_cont_alone_none" "NONE" "none" "continuous_lc_alone" \
    "$LC_SERVER_PARALLEL" 1 "$LC_CONCURRENCY" 1

  start_lc_server "${case_dir}/lc_events" "NONE" "$LC_SERVER_PARALLEL"
  start_ebpf_trace "$case_dir"
  warmup_client "$LC_PORT" "$N_WARMUP_LC" "LC"

  run_client_concurrent "$LC_PORT" "lc" "$N_LC" "${case_dir}/lc_client.jsonl" "$LC_CONCURRENCY"

  stop_ebpf_trace
  check_case_events "$case_dir" "$(( 2 * N_LC ))"
  ebpf_quick_summary "$case_dir"
  cleanup_servers
}

run_case_lc_cont_be_long_none() {
  local case_dir="${OUT_ROOT}/caseJ_lc_cont_be_long_none"
  mkdir -p "$case_dir"

  echo "===== CASE J: continuous LC + BE-long, no policy, LC-pool-first ====="
  reset_shared_memory
  save_config "$case_dir" "lc_cont_be_long_none" "NONE" "LONG" "lc_pool_first" \
    "$LC_SERVER_PARALLEL" "$BE_SERVER_PARALLEL" "$LC_CONCURRENCY" 1

  start_lc_server "${case_dir}/lc_events" "NONE" "$LC_SERVER_PARALLEL"
  start_be_server "${case_dir}/be_events" "NONE" "LONG" "$BE_SERVER_PARALLEL"
  start_ebpf_trace "$case_dir"

  warmup_client "$LC_PORT" "$N_WARMUP_LC" "LC"
  warmup_client "$BE_PORT" "$N_WARMUP_BE" "BE"

  run_client_concurrent "$LC_PORT" "lc" "$N_LC" "${case_dir}/lc_client.jsonl" "$LC_CONCURRENCY" &
  local c_lc=$!

  sleep "$LC_BE_LAUNCH_GAP_SEC"

  run_client "$BE_PORT" "be_long" "$N_BE_LONG" "${case_dir}/be_client.jsonl" &
  local c_be=$!

  wait "$c_lc"
  wait "$c_be"

  stop_ebpf_trace
  check_case_events "$case_dir" "$(( 2 * (N_LC + N_BE_LONG) ))"
  ebpf_quick_summary "$case_dir"
  cleanup_servers
}

run_case_lc_cont_be_long_policy() {
  local case_dir="${OUT_ROOT}/caseK_lc_cont_be_long_policy"
  mkdir -p "$case_dir"

  echo "===== CASE K: continuous LC + BE-long, CAP policy, LC-pool-first ====="
  reset_shared_memory
  save_config "$case_dir" "lc_cont_be_long_policy" "CAP" "LONG" "lc_pool_first" \
    "$LC_SERVER_PARALLEL" "$BE_SERVER_PARALLEL" "$LC_CONCURRENCY" 1

  start_lc_server "${case_dir}/lc_events" "CAP" "$LC_SERVER_PARALLEL"
  start_be_server "${case_dir}/be_events" "CAP" "LONG" "$BE_SERVER_PARALLEL"
  start_ebpf_trace "$case_dir"

  warmup_client "$LC_PORT" "$N_WARMUP_LC" "LC"
  warmup_client "$BE_PORT" "$N_WARMUP_BE" "BE"

  run_client_concurrent "$LC_PORT" "lc" "$N_LC" "${case_dir}/lc_client.jsonl" "$LC_CONCURRENCY" &
  local c_lc=$!

  sleep "$LC_BE_LAUNCH_GAP_SEC"

  run_client "$BE_PORT" "be_long" "$N_BE_LONG" "${case_dir}/be_client.jsonl" &
  local c_be=$!

  wait "$c_lc"
  wait "$c_be"

  stop_ebpf_trace
  check_case_events "$case_dir" "$(( 2 * (N_LC + N_BE_LONG) ))"
  ebpf_quick_summary "$case_dir"
  cleanup_servers
}

run_base_cases() {
  run_case_lc_alone
  run_case_be_long_alone
  run_case_lc_be_long_none_be_first
  run_case_lc_be_long_policy_be_first
  run_case_lc_be_short_none
  run_case_lc_be_short_policy
  run_case_lc_first_be_long_none
  run_case_lc_first_be_long_policy
}

run_continuous_cases() {
  run_case_lc_cont_alone_none
  run_case_lc_cont_be_long_none
  run_case_lc_cont_be_long_policy
}

# ------------------------------------------------------------------------------
# Letter-to-function dispatch (new)
# ------------------------------------------------------------------------------
run_case_by_letter() {
  case "$1" in
    A) run_case_lc_alone ;;
    B) run_case_be_long_alone ;;
    C) run_case_lc_be_long_none_be_first ;;
    D) run_case_lc_be_long_policy_be_first ;;
    E) run_case_lc_be_short_none ;;
    F) run_case_lc_be_short_policy ;;
    G) run_case_lc_first_be_long_none ;;
    H) run_case_lc_first_be_long_policy ;;
    I) run_case_lc_cont_alone_none ;;
    J) run_case_lc_cont_be_long_none ;;
    K) run_case_lc_cont_be_long_policy ;;
    *) echo "ERROR: unknown case letter '$1'" >&2; exit 1 ;;
  esac
}

# One replicate's worth of work. Respects CASES if set, else RUN_SET.
dispatch_one_replicate() {
  if [[ -n "$CASES" ]]; then
    for letter in $CASES; do
      run_case_by_letter "$letter"
    done
  else
    case "$RUN_SET" in
      all)        run_base_cases; run_continuous_cases ;;
      base)       run_base_cases ;;
      continuous) run_continuous_cases ;;
      *)
        echo "ERROR: unknown RUN_SET '$RUN_SET'. Use: all, base, or continuous." >&2
        exit 1
        ;;
    esac
  fi
}

# ------------------------------------------------------------------------------
# Main loop: replicates
# ------------------------------------------------------------------------------
if [[ "$N_REPS" -gt 1 ]]; then
  for rep in $(seq 1 "$N_REPS"); do
    rep_padded=$(printf "rep%02d" "$rep")
    OUT_ROOT="${OUT_ROOT_BASE}/${rep_padded}"
    mkdir -p "$OUT_ROOT"
    echo
    echo "########################################"
    echo "# REPLICATE $rep / $N_REPS"
    echo "# OUT_ROOT: $OUT_ROOT"
    echo "########################################"
    echo
    dispatch_one_replicate
  done
else
  # Single-rep mode preserves the original flat layout.
  OUT_ROOT="$OUT_ROOT_BASE"
  dispatch_one_replicate
fi

echo
echo "All requested runs complete."
echo "Results in: $OUT_ROOT_BASE"

if [[ "$EBPF_TRACE" == "1" ]]; then
  echo
  echo "eBPF per-case quick summary (BE policy-backoff signature):"
  # Handle both layouts: flat (single-rep) and nested (repNN/caseX_*/).
  for d in "$OUT_ROOT_BASE"/case*/ "$OUT_ROOT_BASE"/rep*/case*/; do
    [[ -d "$d" ]] || continue
    ebpf_quick_summary "$d"
  done
fi
