#!/usr/bin/env bash
set -euo pipefail

OUT="./out/output_week2_warm_clean"
PARSER="./parser_week2_full.py"

if [[ ! -d "$OUT" ]]; then
    echo "[ERROR] Output directory does not exist: $OUT" >&2
    exit 1
fi

if [[ ! -f "$PARSER" ]]; then
    echo "[ERROR] Parser not found: $PARSER" >&2
    echo "Copy parser_week2_full_clean.py into the current working directory first." >&2
    exit 1
fi

mkdir -p "$OUT/analysis"

echo
echo "======================================================================"
echo "[ANALYSIS] Running parser on warm-clean dataset"
echo "======================================================================"

python3 "$PARSER" "$OUT" --quiet-runs \
    | tee "$OUT/analysis/parser_terminal_output.txt"

echo
echo "======================================================================"
echo "[ANALYSIS] Generated files"
echo "======================================================================"

ls -lh "$OUT/analysis"

echo
echo "Key CSVs:"
echo "  $OUT/analysis/lc_request_summary.csv"
echo "  $OUT/analysis/lc_phase_summary.csv"
echo "  $OUT/analysis/be_work_summary.csv"
echo "  $OUT/analysis/overlap_summary.csv"
echo
echo "Terminal output saved to:"
echo "  $OUT/analysis/parser_terminal_output.txt"

