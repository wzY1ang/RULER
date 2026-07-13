#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INPUT_PATH="${INPUT_PATH:?Please set INPUT_PATH to the raw LeCaRD JSON/JSONL path}"
OUTPUT_PATH="${OUTPUT_PATH:?Please set OUTPUT_PATH to the filtered subset output path}"
STATS_OUTPUT_PATH="${STATS_OUTPUT_PATH:-}"
MIN_LABELS="${MIN_LABELS:-3}"
TOP_K="${TOP_K:-20}"

CMD=(
  python "$REPO_ROOT/check_data.py"
  --input_path "$INPUT_PATH"
  --output_path "$OUTPUT_PATH"
  --min_labels "$MIN_LABELS"
  --top_k "$TOP_K"
)

if [[ -n "$STATS_OUTPUT_PATH" ]]; then
  CMD+=(--stats_output_path "$STATS_OUTPUT_PATH")
fi

"${CMD[@]}"
