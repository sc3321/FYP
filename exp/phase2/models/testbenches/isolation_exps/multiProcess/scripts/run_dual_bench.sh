#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  $0 --bin1 <path> --bin2 <path> --out <base_dir> \
     [--mode1 <large|medium|small|tiny|micro>] \
     [--sync1 <per_iter|final|none>] \
     [--iters1 <n>] [--rows1 <n>] [--cols1 <n>] [--cpu1 <core>] \
     [--mode2 <large|medium|small|tiny|micro>] \
     [--sync2 <per_iter|final|none>] \
     [--iters2 <n>] [--rows2 <n>] [--cols2 <n>] [--cpu2 <core>] \
     [--label <name>]

Example:
  $0 --bin1 ./bench --bin2 ./bench --out runs/noMPS \
     --mode1 small --sync1 per_iter --iters1 500 --rows1 512 --cols1 512 \
     --mode2 large --sync2 per_iter --iters2 500 --rows2 512 --cols2 512 \
     --cpu1 0 --cpu2 1 \
     --label small_vs_large_periter_512
EOF
}

BIN1=""
BIN2=""
OUT_BASE=""
LABEL=""

MODE1="large"
SYNC1="per_iter"
ITERS1="500"
ROWS1="512"
COLS1="512"
CPU1="0"

MODE2="large"
SYNC2="per_iter"
ITERS2="500"
ROWS2="512"
COLS2="512"
CPU2="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin1) BIN1="$2"; shift 2 ;;
    --bin2) BIN2="$2"; shift 2 ;;
    --out) OUT_BASE="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;

    --mode1) MODE1="$2"; shift 2 ;;
    --sync1) SYNC1="$2"; shift 2 ;;
    --iters1) ITERS1="$2"; shift 2 ;;
    --rows1) ROWS1="$2"; shift 2 ;;
    --cols1) COLS1="$2"; shift 2 ;;
    --cpu1) CPU1="$2"; shift 2 ;;

    --mode2) MODE2="$2"; shift 2 ;;
    --sync2) SYNC2="$2"; shift 2 ;;
    --iters2) ITERS2="$2"; shift 2 ;;
    --rows2) ROWS2="$2"; shift 2 ;;
    --cols2) COLS2="$2"; shift 2 ;;
    --cpu2) CPU2="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$BIN1" || -z "$BIN2" || -z "$OUT_BASE" ]]; then
  echo "Missing required arguments."
  usage
  exit 1
fi

if [[ ! -x "$BIN1" ]]; then
  echo "BIN1 is not executable: $BIN1"
  exit 1
fi

if [[ ! -x "$BIN2" ]]; then
  echo "BIN2 is not executable: $BIN2"
  exit 1
fi

# Build an automatic label if not supplied.
if [[ -z "$LABEL" ]]; then
  LABEL="P1-${MODE1}_${SYNC1}_${ITERS1}_${ROWS1}x${COLS1}__P2-${MODE2}_${SYNC2}_${ITERS2}_${ROWS2}x${COLS2}"
fi

RUN_DIR="${OUT_BASE}/${LABEL}"
P1_DIR="${RUN_DIR}/P1"
P2_DIR="${RUN_DIR}/P2"

mkdir -p "$P1_DIR" "$P2_DIR"

STRACE_OUT_P1="${P1_DIR}/strace"
STRACE_OUT_P2="${P2_DIR}/strace"

CONFIG_P1="${P1_DIR}/config.txt"
CONFIG_P2="${P2_DIR}/config.txt"
META_FILE="${RUN_DIR}/meta.txt"

CMD1=(taskset -c "$CPU1" "$BIN1" "$MODE1" "$SYNC1" "$ITERS1" "$ROWS1" "$COLS1")
CMD2=(taskset -c "$CPU2" "$BIN2" "$MODE2" "$SYNC2" "$ITERS2" "$ROWS2" "$COLS2")

{
  echo "run_dir=${RUN_DIR}"
  echo "label=${LABEL}"
  echo "bin1=${BIN1}"
  echo "bin2=${BIN2}"
  echo "cpu1=${CPU1}"
  echo "cpu2=${CPU2}"
  echo "cmd1=${CMD1[*]}"
  echo "cmd2=${CMD2[*]}"
  echo "start_wall=$(date --iso-8601=seconds)"
} > "$META_FILE"

{
  echo "binary=${BIN1}"
  echo "mode=${MODE1}"
  echo "sync=${SYNC1}"
  echo "iters=${ITERS1}"
  echo "rows=${ROWS1}"
  echo "cols=${COLS1}"
  echo "cpu=${CPU1}"
  echo "cmd=${CMD1[*]}"
} > "$CONFIG_P1"

{
  echo "binary=${BIN2}"
  echo "mode=${MODE2}"
  echo "sync=${SYNC2}"
  echo "iters=${ITERS2}"
  echo "rows=${ROWS2}"
  echo "cols=${COLS2}"
  echo "cpu=${CPU2}"
  echo "cmd=${CMD2[*]}"
} > "$CONFIG_P2"

echo "Launching P1:"
echo "  ${CMD1[*]}"
echo "Launching P2:"
echo "  ${CMD2[*]}"
echo "Output dir:"
echo "  ${RUN_DIR}"

# Use time so each process runtime is recorded.
(
  /usr/bin/time -f "real=%e user=%U sys=%S" -o "${P1_DIR}/time.txt" \
    strace -ff -ttt -T \
      -e trace=ioctl,write,mmap,munmap,mprotect,futex,poll,ppoll,epoll_wait,clone,clone3 \
      -o "$STRACE_OUT_P1" \
      "${CMD1[@]}"
) &
PID1=$!

(
  /usr/bin/time -f "real=%e user=%U sys=%S" -o "${P2_DIR}/time.txt" \
    strace -ff -ttt -T \
      -e trace=ioctl,write,mmap,munmap,mprotect,futex,poll,ppoll,epoll_wait,clone,clone3 \
      -o "$STRACE_OUT_P2" \
      "${CMD2[@]}"
) &
PID2=$!

wait "$PID1"
wait "$PID2"

echo "end_wall=$(date --iso-8601=seconds)" >> "$META_FILE"
echo "Both processes completed."
echo "Saved in: ${RUN_DIR}"
