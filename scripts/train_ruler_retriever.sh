#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_MODEL_DIR="${BASE_MODEL_DIR:?Please set BASE_MODEL_DIR to the Qwen3 base model path}"
TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/retriever/train/dense_train.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/qwen3_retriever}"

mkdir -p "$OUTPUT_DIR"

CMD=(
  python "$REPO_ROOT/retriever/llm2vec_lasttoken/train_qwen3_full.py"
  --model_name_or_path "$BASE_MODEL_DIR"
  --tokenizer_name "$BASE_MODEL_DIR"
  --train_path "$TRAIN_FILE"
  --output_dir "$OUTPUT_DIR"
  --q_max_len "${Q_MAX_LEN:-512}"
  --p_max_len "${P_MAX_LEN:-200}"
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE:-4}"
  --learning_rate "${LEARNING_RATE:-1e-5}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-5}"
  --save_strategy "${SAVE_STRATEGY:-epoch}"
)

if [[ -n "${MAX_STEPS:-}" ]]; then
  CMD+=(--max_steps "$MAX_STEPS")
fi

"${CMD[@]}"
