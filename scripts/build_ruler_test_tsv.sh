#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TEST_JSON="${TEST_JSON:?Please set TEST_JSON to the test case JSON/JSONL path}"
LAW_CORPUS="${LAW_CORPUS:?Please set LAW_CORPUS to the law corpus JSONL path}"
RANK_TEST="${RANK_TEST:?Please set RANK_TEST to the test ranking TSV path}"
OUT_FILE="${OUT_FILE:?Please set OUT_FILE to the grouped test TSV output path}"
TOPK="${TOPK:-50}"

python "$REPO_ROOT/build_test_dataset.py" \
  --test_json "$TEST_JSON" \
  --law_corpus "$LAW_CORPUS" \
  --rank_test "$RANK_TEST" \
  --out_file "$OUT_FILE" \
  --topk "$TOPK"
