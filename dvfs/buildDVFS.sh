#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Build script for the DVFS daemon and the power/clock test binaries.
#
# Place at:  ~/FYP/dvfs/build_dvfs.sh
# Usage:
#   ./build_dvfs.sh              # build daemon + test binaries
#   ./build_dvfs.sh daemon       # build daemon only
#   ./build_dvfs.sh test         # build test binaries only
#   ./build_dvfs.sh clean        # remove bin/
#
# Outputs to  dvfs/bin/
# ==============================================================================

DVFS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FYP_ROOT="$(cd "${DVFS_DIR}/.." && pwd)"

PHASEGUARD="${FYP_ROOT}/phaseGuard"
PHASEGUARD_LLAMA="${FYP_ROOT}/llama_validation/phaseGuard"
BIN_DIR="${DVFS_DIR}/bin"

TARGET="${1:-all}"

if [[ "$TARGET" == "clean" ]]; then
  rm -rf "$BIN_DIR"
  echo "Removed ${BIN_DIR}"
  exit 0
fi

# ------------------------------------------------------------------------------
# Locate nvml.h. It ships with the CUDA toolkit, not the driver package, so it
# lives somewhere different from libnvidia-ml.so.
# ------------------------------------------------------------------------------
NVML_INC=""
for d in \
    /usr/local/cuda/include \
    /usr/local/cuda-12.7/include \
    /usr/local/cuda-12.6/include \
    /usr/include/nvidia/gdk \
    /usr/include ; do
  if [[ -f "${d}/nvml.h" ]]; then
    NVML_INC="$d"
    break
  fi
done

if [[ -z "$NVML_INC" ]]; then
  echo "ERROR: could not find nvml.h." >&2
  echo "       Locate it with:  find / -name nvml.h 2>/dev/null" >&2
  exit 1
fi

# ------------------------------------------------------------------------------
# Locate something the linker can actually resolve for -lnvidia-ml.
#
# The driver package installs libnvidia-ml.so.1 (versioned, runtime). The
# unversioned libnvidia-ml.so that -l needs comes from a -dev package or from
# the CUDA toolkit's stubs directory. nvcc adds the stubs path implicitly,
# which is why nvcc links this fine and plain g++ does not.
#
# Linking the stub is correct, not a workaround: the stub carries the same
# SONAME (libnvidia-ml.so.1), so at runtime the loader binds the real driver
# library found via ldconfig.
# ------------------------------------------------------------------------------
NVML_LINK=()
NVML_SRC=""

# 1. A real unversioned .so, if a -dev package provided one.
for d in /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib /usr/local/cuda/lib64 ; do
  if [[ -e "${d}/libnvidia-ml.so" ]]; then
    NVML_LINK=(-L"${d}" -lnvidia-ml)
    NVML_SRC="${d}/libnvidia-ml.so (unversioned)"
    break
  fi
done

# 2. CUDA toolkit stubs.
if [[ ${#NVML_LINK[@]} -eq 0 ]]; then
  for d in \
      /usr/local/cuda/lib64/stubs \
      /usr/local/cuda-12.7/lib64/stubs \
      /usr/local/cuda-12.6/lib64/stubs ; do
    if [[ -e "${d}/libnvidia-ml.so" ]]; then
      NVML_LINK=(-L"${d}" -lnvidia-ml)
      NVML_SRC="${d}/libnvidia-ml.so (CUDA stub)"
      break
    fi
  done
fi

# 3. Link the versioned driver library directly by absolute path.
if [[ ${#NVML_LINK[@]} -eq 0 ]]; then
  cand="$(ldconfig -p | awk '/libnvidia-ml\.so\.1/ {print $NF; exit}')"
  if [[ -n "${cand:-}" && -e "$cand" ]]; then
    NVML_LINK=("$cand")
    NVML_SRC="${cand} (versioned, absolute path)"
  fi
fi

if [[ ${#NVML_LINK[@]} -eq 0 ]]; then
  echo "ERROR: could not find any linkable NVML library." >&2
  echo "       Searched for libnvidia-ml.so and CUDA stubs, and for" >&2
  echo "       libnvidia-ml.so.1 via ldconfig." >&2
  echo "       Diagnose with:  find / -name 'libnvidia-ml.so*' 2>/dev/null" >&2
  exit 1
fi

# ------------------------------------------------------------------------------
# Guard against divergence between the two phaseGuard trees.
#
# The daemon includes ${PHASEGUARD}/include/memManager.h and links
# ${PHASEGUARD}/src/policy/memManager.cpp. The llama servers build against
# ${PHASEGUARD_LLAMA}. If policyData's layout differs between the two, the
# daemon misreads shared memory with no error at any level: no compile error,
# no link error, no runtime error, just wrong counter values.
# ------------------------------------------------------------------------------
if [[ -f "${PHASEGUARD_LLAMA}/include/memManager.h" ]]; then
  if ! diff -q "${PHASEGUARD}/include/memManager.h" \
                "${PHASEGUARD_LLAMA}/include/memManager.h" >/dev/null; then
    echo "ERROR: memManager.h differs between the two phaseGuard trees." >&2
    echo "       policyData's layout must be identical or the daemon will" >&2
    echo "       silently misread shared memory. Reconcile first:" >&2
    echo "         diff ${PHASEGUARD}/include/memManager.h ${PHASEGUARD_LLAMA}/include/memManager.h" >&2
    exit 1
  fi
  echo "phaseGuard layout check:  OK (both trees identical)"
else
  echo "WARNING: ${PHASEGUARD_LLAMA}/include/memManager.h not found; skipping layout check." >&2
fi

mkdir -p "$BIN_DIR"

echo "FYP root:                ${FYP_ROOT}"
echo "nvml.h:                  ${NVML_INC}"
echo "NVML link:               ${NVML_SRC}"
echo "Output:                  ${BIN_DIR}"
echo ""

CXXFLAGS=(-std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter)

build_daemon() {
  echo "--- dvfsDaemon ---"
  g++ "${CXXFLAGS[@]}" \
      "${DVFS_DIR}/src/dvfsControllerMain.cpp" \
      "${DVFS_DIR}/src/dvfsManager.cpp" \
      "${PHASEGUARD}/src/policy/memManager.cpp" \
      -I"${DVFS_DIR}/include" \
      -I"${PHASEGUARD}/include" \
      -I"${NVML_INC}" \
      "${NVML_LINK[@]}" -lpthread -lrt \
      -o "${BIN_DIR}/dvfsDaemon"
  echo "    -> ${BIN_DIR}/dvfsDaemon"
}

build_test() {
  echo "--- test (power limit setter) ---"
  g++ "${CXXFLAGS[@]}" \
      "${DVFS_DIR}/test/test.cpp" \
      -I"${NVML_INC}" "${NVML_LINK[@]}" \
      -o "${BIN_DIR}/test"
  echo "    -> ${BIN_DIR}/test"

  echo "--- poll (clock sampler) ---"
  g++ "${CXXFLAGS[@]}" \
      "${DVFS_DIR}/test/poll.cpp" \
      -I"${NVML_INC}" "${NVML_LINK[@]}" \
      -o "${BIN_DIR}/poll"
  echo "    -> ${BIN_DIR}/poll"

  echo "--- sustainedLoad (compute-bound generator) ---"
  if command -v nvcc >/dev/null 2>&1; then
    nvcc -O3 -arch=sm_61 \
        "${DVFS_DIR}/test/sustainedLoad.cu" \
        -o "${BIN_DIR}/sustainedLoad"
    echo "    -> ${BIN_DIR}/sustainedLoad  (sm_61, Pascal)"
  else
    echo "    SKIPPED: nvcc not on PATH" >&2
  fi
}

case "$TARGET" in
  all)    build_daemon; echo ""; build_test ;;
  daemon) build_daemon ;;
  test)   build_test ;;
  *)      echo "Unknown target '$TARGET'. Use: all | daemon | test | clean" >&2; exit 1 ;;
esac

echo ""
echo "Build complete."
ls -la "$BIN_DIR"

# ------------------------------------------------------------------------------
# Confirm the daemon can actually resolve NVML at runtime, which is a separate
# question from whether it linked.
# ------------------------------------------------------------------------------
if [[ -x "${BIN_DIR}/dvfsDaemon" ]]; then
  echo ""
  echo "Runtime library resolution for dvfsDaemon:"
  if ldd "${BIN_DIR}/dvfsDaemon" | grep -q "not found"; then
    echo "  WARNING: unresolved shared libraries:" >&2
    ldd "${BIN_DIR}/dvfsDaemon" | grep "not found" >&2
    echo "  If this is the CUDA stub path, the real driver library is missing" >&2
    echo "  from the loader path. Check: ldconfig -p | grep nvidia-ml" >&2
  else
    ldd "${BIN_DIR}/dvfsDaemon" | grep -i nvidia || true
    echo "  OK"
  fi
fi
