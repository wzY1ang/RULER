#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAIN_TSV="${TRAIN_TSV:?Please set TRAIN_TSV to the reranker training TSV path}"
STAGE1_MODEL_PATH="${STAGE1_MODEL_PATH:?Please set STAGE1_MODEL_PATH to the Stage 1 retriever checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/ruler_reranker}"

mkdir -p "$OUTPUT_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

CMD=(
  torchrun --standalone --nproc_per_node "$NPROC_PER_NODE"
  "$REPO_ROOT/retriever/llm2vec_lasttoken/reranker/src/train_lora.py"
  --tsv "$TRAIN_TSV" \
  --model_path "$STAGE1_MODEL_PATH" \
  --out_dir "$OUTPUT_DIR" \
  --epochs "${NUM_TRAIN_EPOCHS:-5}" \
  --batch_groups "${BATCH_GROUPS:-8}" \
  --lr "${LEARNING_RATE:-1e-5}" \
  --q_max "${Q_MAX_LEN:-256}" \
  --d_max "${D_MAX_LEN:-256}" \
  --margin "${MARGIN:-1.0}" \
  --alpha "${ALPHA:-3}" \
  --gamma "${GAMMA:-0.0}" \
  --beta_IRDA "${BETA_IRDA:-0.0}" \
  --beta_ent "${BETA_ENT:-0.2}" \
  --temp "${TEMPERATURE:-1.0}" \
  --lora_r "${LORA_R:-32}" \
  --lora_alpha "${LORA_ALPHA:-64}" \
  --lora_dropout "${LORA_DROPOUT:-0.1}" \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --use_lora
)

if [[ "${GRADIENT_CHECKPOINTING:-1}" == "1" ]]; then
  CMD+=(--grad_checkpoint)
fi

"${CMD[@]}"
