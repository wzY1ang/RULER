#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
QUERY_FILE="${QUERY_FILE:-$DATA_DIR/test.json}"
CORPUS_FILE="${CORPUS_FILE:-$DATA_DIR/law_corpus.jsonl}"
RERANK_TSV="${RERANK_TSV:?Please set RERANK_TSV to the reranker evaluation TSV path}"

RETRIEVER_MODEL_PATH="${RETRIEVER_MODEL_PATH:?Please set RETRIEVER_MODEL_PATH to the Stage 1 retriever checkpoint}"
RERANKER_MODEL_PATH="${RERANKER_MODEL_PATH:?Please set RERANKER_MODEL_PATH to the Stage 2 reranker checkpoint or LoRA adapter path}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:?Please set BASE_MODEL_PATH to the Qwen3 base model path}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/benchmark_ruler}"
ENCODE_DIR="$OUTPUT_DIR/encode"
mkdir -p "$ENCODE_DIR"

python "$REPO_ROOT/retriever/llm2vec_lasttoken/encode_with_qwen3.py" \
  --model_name_or_path "$RETRIEVER_MODEL_PATH" \
  --tokenizer_name "$BASE_MODEL_PATH" \
  --encode_in_path "$QUERY_FILE" \
  --encoded_save_path "$ENCODE_DIR/query.pt" \
  --q_max_len "${Q_MAX_LEN:-512}" \
  --batch_size "${ENCODE_BATCH_SIZE:-64}"

python "$REPO_ROOT/retriever/llm2vec_lasttoken/encode_with_qwen3.py" \
  --model_name_or_path "$RETRIEVER_MODEL_PATH" \
  --tokenizer_name "$BASE_MODEL_PATH" \
  --encode_in_path "$CORPUS_FILE" \
  --encoded_save_path "$ENCODE_DIR/corpus.pt" \
  --p_max_len "${D_MAX_LEN:-200}" \
  --batch_size "${ENCODE_BATCH_SIZE:-64}"

(
  cd "$REPO_ROOT/retriever"
  python -m dense.faiss_retriever \
    --query_reps "$ENCODE_DIR/query.pt" \
    --passage_reps "$ENCODE_DIR/corpus.pt" \
    --depth "${RETRIEVAL_DEPTH:-100}" \
    --batch_size -1 \
    --save_text \
    --save_ranking_to "$OUTPUT_DIR/rank.tsv"
)

python "$REPO_ROOT/retriever/llm2vec_lasttoken/reranker/src/eval_deep_ours.py" \
  --tsv "$RERANK_TSV" \
  --model_path "$RERANKER_MODEL_PATH" \
  --base_model_path "$RETRIEVER_MODEL_PATH" \
  --q_max "${Q_MAX_LEN:-512}" \
  --d_max "${D_MAX_LEN:-200}" \
  --batch_groups "${RERANK_BATCH_GROUPS:-8}" \
  --cutoff "${RERANK_CUTOFF:-10}" \
  --output_dir "$OUTPUT_DIR/reranker_metrics"
