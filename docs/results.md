# Reported Results

These values are transcribed from the SIGIR 2026 paper and are provided as
reference results.

## Retrieval

| Dataset | MRR@100 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| JuDGE-Stat | 0.8965 | 0.6570 | 0.8150 |
| LeCaRDv2-Stat | 0.9001 | 0.6832 | 0.8350 |

The paper uses a FAISS dense index, retrieves Top-100 documents, and retains
Top-50 candidates for Stage 2.

## Reranking

| Dataset | NDCG@10 | MAP@10 | R-Prec | MRR@10 |
|---|---:|---:|---:|---:|
| JuDGE-Stat | 0.9013 | 0.8299 | 0.7672 | 0.9693 |
| LeCaRDv2-Stat | 0.8605 | 0.7653 | 0.7157 | 0.9498 |

## Robustness

The paper reports NR@R and score-distribution overlap for diagnosing phantom
hits. `retriever/llm2vec_lasttoken/reranker/src/calc_dist_metrics.py` contains
the distribution-metric implementation.

Dataset artifacts and model checkpoints are not redistributed in this
repository.
