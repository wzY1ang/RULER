# Data Guide

Full datasets are not committed to this code repository. The public dataset
page is [RULER-dataset/RULER](https://huggingface.co/datasets/RULER-dataset/RULER).
Users must also follow the redistribution terms of the source datasets.

## Local Layout

```text
data/
|-- train.json
|-- test.json
|-- law_corpus.jsonl
|-- qrels_file_train
`-- qrels_file_test
```

## Query Format

`train.json` and `test.json` may be JSON arrays or JSONL files. Each record uses:

```json
{
  "text_id": "query-id",
  "text": "query text",
  "la": ["positive-document-id"]
}
```

The aliases `qid`, `query`, and `positives` are also accepted.

## Corpus Format

`law_corpus.jsonl` contains one document per line:

```json
{"text_id": "document-id", "name": "optional title", "text": "document text"}
```

The alias `doc_id` is accepted for `text_id`.

## Stage 1 Training Format

`retriever/train/dense_train.json` is JSONL with one record per query:

```json
{
  "query": "query text",
  "positives": ["positive document text"],
  "negatives": ["negative document text"]
}
```

## Ranking Format

Stage 1 rankings are whitespace-separated and contain at least two columns:

```text
query-id document-id score
```

The score is optional for the group builders. The first two columns are used.

## Grouped TSV Format

The Stage 2 builders output:

```text
query_id query cand_id cand_text group_id cand_label group_label
```

Fields are tab-separated. `cand_label` marks document relevance and
`group_label` marks whether the group contains any positive candidate.
