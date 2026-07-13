#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAIN_JSON="${TRAIN_JSON:?Please set TRAIN_JSON to the training case JSON/JSONL path}"
LAW_CORPUS="${LAW_CORPUS:?Please set LAW_CORPUS to the law corpus JSONL path}"
RANK_TRAIN="${RANK_TRAIN:?Please set RANK_TRAIN to the training ranking TSV path}"
OUT_FILE="${OUT_FILE:?Please set OUT_FILE to the grouped training TSV output path}"

SAMPLE_N="${SAMPLE_N:-}"

CMD=(
  python "$REPO_ROOT/build_train_dataset.py"
  --train_json "$TRAIN_JSON"
  --law_corpus "$LAW_CORPUS"
  --rank_train "$RANK_TRAIN"
  --out_file "$OUT_FILE"
)

if [[ -n "$SAMPLE_N" ]]; then
  CMD+=(--sample_n "$SAMPLE_N")
fi

"${CMD[@]}"
