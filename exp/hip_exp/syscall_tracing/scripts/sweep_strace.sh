#!/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"



# Usage: sh sweep_strace.sh <alloc|baseline|kernels>
BIN="${1:-}"
if [ -z "$BIN" ]; then
  echo "Usage: $0 <alloc|baseline|kernels>" >&2
  exit 2
fi

EXE="$ROOT_DIR/build/$BIN"
if [ ! -x "$EXE" ]; then
  echo "Error: not found or not executable: $EXE" >&2
  exit 3
fi

# Defaults (override via env vars)
BYTES="${BYTES:-1000000 5000000 10000000}"
ITERS="${ITERS:-100 5000 10000 50000 100000}"
REPEATS="${REPEATS:-1}"
WARMUP="${WARMUP:-1}"

RESULTS_ROOT="$ROOT/FYP/exp/hip_exp/syscall_tracing/results"
if [ ! -d "$RESULTS_ROOT" ]; then
    # Directory does not exist, so create it
    mkdir "$RESULTS_ROOT"

fi
RUN_ID="$(date +"%Y%m%d_%H%M%S")"
OUTDIR="$RESULTS_ROOT/$BIN/run_$RUN_ID"
mkdir -p "$OUTDIR"

echo "== $BIN sweep =="
echo "exe: $EXE"
echo "out: $OUTDIR"
echo "bytes: $BYTES"
echo "iters: $ITERS"
echo "warmup: $WARMUP, repeats: $REPEATS"
echo ""

# Warmup (no strace): use first byte/iter values
if [ "$WARMUP" -gt 0 ]; then
  b0=$(printf "%s\n" "$BYTES" | awk '{print $1}')
  i0=$(printf "%s\n" "$ITERS" | awk '{print $1}')
  w=1
  while [ "$w" -le "$WARMUP" ]; do
    "$EXE" "$b0" "$i0" >/dev/null 2>&1 || true
    w=$((w + 1))
  done
fi

# Sweep with strace -c
for b in $BYTES; do
  for it in $ITERS; do
    r=1
    while [ "$r" -le "$REPEATS" ]; do
      out="$OUTDIR/${BIN}_${b}b_${it}iters_rep${r}.txt"
      echo "strace -c $BIN bytes=$b iters=$it rep=$r"
      strace -qq -c -o "$out" "$EXE" "$b" "$it" >/dev/null 2>&1 || {
        echo "  !! failed: $EXE $b $it (continuing)" >&2
      }
      r=$((r + 1))
    done
  done
done

# Update latest symlink (ignore failure on filesystems that don't support symlinks)
ln -sfn "run_$RUN_ID" "$RESULTS_ROOT/$BIN/latest" 2>/dev/null || true

echo ""
echo "Done."
echo "Results: $OUTDIR"
echo "Latest:  $RESULTS_ROOT/$BIN/latest"


