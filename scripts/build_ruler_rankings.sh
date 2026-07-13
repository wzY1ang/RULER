#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

QUERY_FILE="${QUERY_FILE:?Please set QUERY_FILE to the query JSON/JSONL path}"
CORPUS_FILE="${CORPUS_FILE:?Please set CORPUS_FILE to the corpus JSONL path}"
RETRIEVER_MODEL_PATH="${RETRIEVER_MODEL_PATH:?Please set RETRIEVER_MODEL_PATH to the Stage 1 retriever checkpoint}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:?Please set BASE_MODEL_PATH to the Qwen3 base model path}"
OUTPUT_RANK_PATH="${OUTPUT_RANK_PATH:?Please set OUTPUT_RANK_PATH to the ranking TSV output path}"

WORK_DIR="${WORK_DIR:-$REPO_ROOT/output/ranking_work}"
Q_MAX_LEN="${Q_MAX_LEN:-512}"
D_MAX_LEN="${D_MAX_LEN:-200}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-64}"
RETRIEVAL_DEPTH="${RETRIEVAL_DEPTH:-100}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:--1}"

mkdir -p "$WORK_DIR"

QUERY_REPS_PATH="$WORK_DIR/query.pt"
CORPUS_REPS_PATH="$WORK_DIR/corpus.pt"

python "$REPO_ROOT/retriever/llm2vec_lasttoken/encode_with_qwen3.py" \
  --model_name_or_path "$RETRIEVER_MODEL_PATH" \
  --tokenizer_name "$BASE_MODEL_PATH" \
  --encode_in_path "$QUERY_FILE" \
  --encoded_save_path "$QUERY_REPS_PATH" \
  --q_max_len "$Q_MAX_LEN" \
  --batch_size "$ENCODE_BATCH_SIZE"

python "$REPO_ROOT/retriever/llm2vec_lasttoken/encode_with_qwen3.py" \
  --model_name_or_path "$RETRIEVER_MODEL_PATH" \
  --tokenizer_name "$BASE_MODEL_PATH" \
  --encode_in_path "$CORPUS_FILE" \
  --encoded_save_path "$CORPUS_REPS_PATH" \
  --p_max_len "$D_MAX_LEN" \
  --batch_size "$ENCODE_BATCH_SIZE"

(
  cd "$REPO_ROOT/retriever"
  python -m dense.faiss_retriever \
    --query_reps "$QUERY_REPS_PATH" \
    --passage_reps "$CORPUS_REPS_PATH" \
    --depth "$RETRIEVAL_DEPTH" \
    --batch_size "$RETRIEVAL_BATCH_SIZE" \
    --save_text \
    --save_ranking_to "$OUTPUT_RANK_PATH"
)
