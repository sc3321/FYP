#!/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Optional arg: restrict to one backend: cpu | cuda
ONLY="${1:-}"

# Executables
CPU_EXE="$ROOT_DIR/build/bin/tb_cpu"
CUDA_EXE="$ROOT_DIR/build/bin/tb_cuda"

# Defaults (override via env vars)
BYTES="${BYTES:-1000000 2500000 5000000 7500000 10000000}"
ITERS="${ITERS:-100 5000 10000 50000 100000}"
REPEATS="${REPEATS:-1}"
WARMUP="${WARMUP:-1}"
VARIANTS="${VARIANTS:-alloc baseline kernels}"

RESULTS_ROOT="$ROOT_DIR/results"
mkdir -p "$RESULTS_ROOT"

RUN_ID="$(date +"%Y%m%d_%H%M%S")"

pick_first() {
  printf "%s\n" "$1" | awk '{print $1}'
}

warmup() {
  exe="$1"
  b0="$(pick_first "$BYTES")"
  i0="$(pick_first "$ITERS")"
  v0="$(pick_first "$VARIANTS")"

  if [ "$WARMUP" -gt 0 ]; then
    w=1
    while [ "$w" -le "$WARMUP" ]; do
      "$exe" "$b0" "$i0" "$v0" >/dev/null 2>&1 || true
      w=$((w + 1))
    done
  fi
}

run_sweep_for() {
  bin_name="$1"
  exe="$2"

  if [ ! -x "$exe" ]; then
    echo "Note: skipping (not found or not executable): $exe"
    return 0
  fi

  OUTDIR="$RESULTS_ROOT/$bin_name/run_$RUN_ID"
  mkdir -p "$OUTDIR"

  echo "== $bin_name sweep =="
  echo "exe: $exe"
  echo "out: $OUTDIR"
  echo "bytes: $BYTES"
  echo "iters: $ITERS"
  echo "variants: $VARIANTS"
  echo "warmup: $WARMUP, repeats: $REPEATS"
  echo ""

  warmup "$exe"

  for v in $VARIANTS; do
    for b in $BYTES; do
      for it in $ITERS; do
        r=1
        while [ "$r" -le "$REPEATS" ]; do
          out="$OUTDIR/${v}_${b}b_${it}iters_rep${r}.txt"
          echo "strace -c $bin_name variant=$v bytes=$b iters=$it rep=$r"
          strace -qq -c -o "$out" "$exe" "$b" "$it" "$v" >/dev/null 2>&1 || {
            echo "  !! failed: $exe $b $it $v (continuing)" >&2
          }
          r=$((r + 1))
        done
      done
    done
  done

  ln -sfn "run_$RUN_ID" "$RESULTS_ROOT/$bin_name/latest" 2>/dev/null || true

  echo ""
  echo "Done."
  echo "Results: $OUTDIR"
  echo "Latest:  $RESULTS_ROOT/$bin_name/latest"
  echo ""
}

if [ -z "$ONLY" ]; then
  run_sweep_for "tb_cpu"  "$CPU_EXE"
  run_sweep_for "tb_cuda" "$CUDA_EXE"
elif [ "$ONLY" = "cpu" ]; then
  run_sweep_for "tb_cpu"  "$CPU_EXE"
elif [ "$ONLY" = "cuda" ]; then
  run_sweep_for "tb_cuda" "$CUDA_EXE"
else
  echo "Usage: $0 [cpu|cuda]" >&2
  exit 2
fi
