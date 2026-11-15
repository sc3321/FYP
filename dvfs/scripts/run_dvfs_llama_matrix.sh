#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Run the llama.cpp phase matrix with the DVFS daemon active.
#
# Place at:  ~/FYP/dvfs/scripts/run_dvfs_llama_matrix.sh
#
# Usage:
#   ./run_dvfs_llama_matrix.sh                       # 1 run, 'continuous' set (3 cases)
#   ./run_dvfs_llama_matrix.sh 1 all                 # 1 run, full A-K matrix (11 cases)
#   ./run_dvfs_llama_matrix.sh 3 continuous          # 3 repeats
#   DVFS_ENABLE=0 ./run_dvfs_llama_matrix.sh 1 all   # control run, DVFS off
#
# This does not duplicate run_llama_phase_matrix.sh. It sets up the environment,
# caches sudo, guarantees the power limit is restored on exit, then delegates.
# The per-case daemon lifecycle lives inside run_llama_phase_matrix.sh.
#
# DVFS output:  dvfs/runs/<matrix_run_name>/<caseName>/dvfs_log.csv
# ==============================================================================

DVFS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FYP_ROOT="$(cd "${DVFS_DIR}/.." && pwd)"
LLAMA_ROOT="${FYP_ROOT}/llama_validation"

N_RUNS="${1:-1}"
RUN_SET="${2:-continuous}"

export DVFS_ENABLE="${DVFS_ENABLE:-1}"
export DVFS_DAEMON="${DVFS_DAEMON:-${DVFS_DIR}/bin/dvfsDaemon}"
export DVFS_RUN_ROOT="${DVFS_RUN_ROOT:-${DVFS_DIR}/runs}"
export DVFS_COOLDOWN_SEC="${DVFS_COOLDOWN_SEC:-90}"

SINGLE_RUNNER="${LLAMA_ROOT}/scripts/run_llama_phase_matrix.sh"
REPEAT_RUNNER="${LLAMA_ROOT}/scripts/run_x_llama_phase_matrices.sh"

# ------------------------------------------------------------------------------
# Preflight
# ------------------------------------------------------------------------------
if [[ ! -x "$SINGLE_RUNNER" ]]; then
  echo "ERROR: ${SINGLE_RUNNER} not found or not executable" >&2
  echo "       chmod +x ${SINGLE_RUNNER}" >&2
  exit 1
fi

if [[ "$DVFS_ENABLE" == "1" ]]; then
  if [[ ! -x "$DVFS_DAEMON" ]]; then
    echo "ERROR: daemon not built at ${DVFS_DAEMON}" >&2
    echo "       Run: ${DVFS_DIR}/build_dvfs.sh daemon" >&2
    exit 1
  fi

  # Cache sudo credentials up front. Without this, a long matrix run blocks on a
  # password prompt the first time the cache expires mid-run, and the symptom is
  # the daemon silently failing to change the power limit.
fi

mkdir -p "$DVFS_RUN_ROOT"

echo "======================================================================"
echo "DVFS + llama phase matrix"
echo "======================================================================"
echo "DVFS_ENABLE:       $DVFS_ENABLE"
echo "DVFS_DAEMON:       $DVFS_DAEMON"
echo "DVFS_RUN_ROOT:     $DVFS_RUN_ROOT"
echo "DVFS_COOLDOWN_SEC: $DVFS_COOLDOWN_SEC"
echo "N_RUNS:            $N_RUNS"
echo "RUN_SET:           $RUN_SET"
echo "Start:             $(date)"
echo "======================================================================"

# ------------------------------------------------------------------------------
# Restore the default power limit on exit whatever happens. The daemon does this
# itself on SIGTERM, but if it is SIGKILLed or this script dies unexpectedly the
# GPU stays pinned at whatever limit was last set, silently affecting the next
# job on this machine.
# ------------------------------------------------------------------------------
restore_power() {
  if [[ "$DVFS_ENABLE" == "1" ]]; then
    echo ""
    echo "Restoring default GPU power limit (250W)..."
    sudo nvidia-smi -pl 250 >/dev/null 2>&1 || true
    nvidia-smi -q -d POWER | grep -E "Current Power Limit|Default Power Limit" || true
  fi
}
trap restore_power EXIT

# ------------------------------------------------------------------------------
# Delegate
# ------------------------------------------------------------------------------
cd "$LLAMA_ROOT"

if [[ "$N_RUNS" == "1" ]]; then
  "$SINGLE_RUNNER" "$RUN_SET"
else
  if [[ ! -x "$REPEAT_RUNNER" ]]; then
    echo "ERROR: ${REPEAT_RUNNER} not found or not executable" >&2
    exit 1
  fi
  "$REPEAT_RUNNER" "$N_RUNS" "$RUN_SET"
fi

echo ""
echo "======================================================================"
echo "Done: $(date)"
echo "DVFS logs under: ${DVFS_RUN_ROOT}"
echo "======================================================================"
find "$DVFS_RUN_ROOT" -name "dvfs_log.csv" -newermt "-1 day" 2>/dev/null | sort | tail -20
