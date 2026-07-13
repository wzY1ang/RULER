# RULER

**RULER: Robust Unified LLM-based Efficient Retrieval for Legal Information**

<p align="center">
  <a href="https://sigir.org/sigir2026/"><img src="https://img.shields.io/badge/Venue-SIGIR%202026-blue" alt="SIGIR 2026"></a>
  <img src="https://img.shields.io/badge/Status-Code%20Released-brightgreen" alt="Code released">
  <img src="https://img.shields.io/badge/Backbone-Qwen3--0.6B-blueviolet" alt="Qwen3-0.6B">
</p>

**Language:** English | [简体中文](README.zh-CN.md)

This is the official repository for **RULER**, published in the proceedings of
**SIGIR 2026**.

> **Current status.** The Stage 1 retriever, data construction pipeline, Stage 2
> reranker, and evaluation scripts are available. Model checkpoints remain
> subject to a separate release decision.

## Table of Contents

- [News](#news)
- [Paper](#paper)
- [Abstract](#abstract)
- [Overview](#overview)
- [Framework](#framework)
- [Highlights](#highlights)
- [Method](#method)
- [Results](#results)
- [Datasets](#datasets)
- [Availability](#availability)
- [Release Roadmap](#release-roadmap)
- [Repository Structure](#repository-structure)
- [Core Entry Points](#core-entry-points)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [Contact](#contact)

## News

- **2026-07:** RULER was published in the proceedings of SIGIR 2026.
- **2026:** Paper accepted to the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval.
- **2026-07:** The cleaned training, data construction, and evaluation code is available.

## Paper

| Item | Information |
|---|---|
| Title | RULER: Robust Unified LLM-based Efficient Retrieval for Legal Information |
| Authors | Chenyu Hou, Ziyang Wang, Bin Cao, Jiaxing Wang, Tianming Zhang, Tiantian Li |
| Venue | SIGIR 2026 |
| Conference | July 20-24, 2026, Melbourne, VIC, Australia |
| DOI | 10.1145/3805712.3809698 |
| Paper | [ACM Digital Library](https://doi.org/10.1145/3805712.3809698) |

## Abstract

Legal information retrieval demands high precision, yet traditional retrieve-then-rerank pipelines with two separate models suffer from cascading error propagation and knowledge disconnects between stages. RULER addresses these issues with a unified parameter-sharing architecture that integrates efficient bi-encoder retrieval and high-precision cross-encoder reranking within a single Qwen3-0.6B backbone. To mitigate phantom hits, where irrelevant documents receive unreasonably high confidence, RULER introduces distribution-robust data construction with all-negative candidate groups, dynamic margin ranking, and maximum entropy regularization. Experiments on JuDGE-Stat and LeCaRDv2-Stat demonstrate strong retrieval, reranking, and robustness performance.

## Overview

Legal information retrieval requires both high recall and high precision. Standard retrieve-then-rerank systems usually rely on two independent models: a bi-encoder retriever for candidate generation and a cross-encoder reranker for fine-grained ranking. Although effective, this separated design introduces parameter redundancy, deployment complexity, and semantic mismatch between stages.

**RULER** addresses this issue with a unified architecture built on a single **Qwen3-0.6B** backbone. The same backbone is progressively adapted for dense retrieval and LoRA-based cross-encoder reranking, allowing the two stages to share representations while preserving the efficiency of a two-stage pipeline.

The framework is especially designed for legal retrieval scenarios where irrelevant candidates may receive over-confident scores. We refer to these high-scoring false positives as **phantom hits** and explicitly model them during reranker training.

## Framework

<p align="center">
  <img src="assets/ruler-framework.png" alt="RULER framework" width="100%">
</p>

RULER consists of three connected parts: Stage 1 retrieval fine-tuning, retrieval-based data construction, and Stage 2 distribution-aware reranking. The training process explicitly models both mixed positive-negative groups and all-negative zero-recall groups.

## Highlights

- **Single 596M checkpoint:** one Qwen3-0.6B checkpoint supports both retrieval and reranking, avoiding two independent deployed models.
- **Strong large-scale retrieval:** RULER reaches 0.9001 MRR@100 and 0.8350 Recall@10 on LeCaRDv2-Stat.
- **High-precision reranking:** RULER achieves 0.8605 NDCG@10 and 0.7653 MAP@10 on LeCaRDv2-Stat.
- **Robust zero-recall behavior:** RULER reduces NR@R to 9.9% on JuDGE-Stat, mitigating over-confident phantom hits.
- **Distribution-aware training:** mixed groups, all-negative groups, dynamic margin ranking, and entropy regularization are combined in one training pipeline.

## Method

RULER keeps the classic retrieve-then-rerank workflow, but shares the underlying backbone across stages:

```text
Query
  -> Stage 1: Qwen3 bi-encoder retriever
  -> Dense vector search with FAISS
  -> Top-50 candidate statutes
  -> Stage 2: Qwen3 + LoRA cross-encoder reranker
  -> Final ranked statute list
```

### Stage 1: Bi-Encoder Retrieval

The retriever fine-tunes Qwen3-0.6B with last-token pooling and normalized embeddings. It generates dense representations for queries and statutes and retrieves top candidates through FAISS search.

### Stage 2: Cross-Encoder Reranking

The reranker starts from the Stage 1 checkpoint and applies LoRA-based fine-tuning. Training groups are constructed from retrieved candidates:

- **Mixed groups:** include relevant statutes and retrieval-hard negative statutes.
- **All-negative groups:** include only irrelevant candidates, simulating zero-recall retrieval outputs.

RULER combines dynamic margin ranking loss with maximum entropy regularization, encouraging sharper separation in mixed groups and calibrated uncertainty in all-negative groups.

## Results

The following numbers are reported in the accepted paper. Reproduction entry
points are provided under `scripts/`.

### Retrieval Results

| Dataset | MRR@100 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| JuDGE-Stat | 0.8965 | 0.6570 | 0.8150 |
| LeCaRDv2-Stat | 0.9001 | 0.6832 | 0.8350 |

### Reranking Results

| Dataset | NDCG@10 | MAP@10 | R-Prec | MRR@10 |
|---|---:|---:|---:|---:|
| JuDGE-Stat | 0.9013 | 0.8299 | 0.7672 | 0.9693 |
| LeCaRDv2-Stat | 0.8605 | 0.7653 | 0.7157 | 0.9498 |

### Robustness Against Phantom Hits

On JuDGE-Stat, RULER reduces noisy high-confidence rankings compared with representative unified baselines:

| Method | NR@R ↓ | Overlap ↓ | NDCG@10 ↑ |
|---|---:|---:|---:|
| UR2N-RandengT5 | 50.6% | 89.6% | 0.7888 |
| BGE-M3 (Finetuned) | 12.7% | 79.2% | 0.8462 |
| RULER | 9.9% | 66.7% | 0.9013 |

## Datasets

RULER is evaluated on two Chinese legal statute retrieval benchmarks:

| Dataset | Source | Queries | Description |
|---|---|---:|---|
| JuDGE-Stat | JuDGE | 2,505 | Small-sample legal statute retrieval benchmark. |
| LeCaRDv2-Stat | LeCaRDv2 | 39,833 | Refined large-scale subset designed to reduce annotation sparsity and improve structural consistency. |

The dataset page is available on Hugging Face. Data preparation instructions
are documented in [`docs/data.md`](docs/data.md). Users should also follow the
usage terms of the original JuDGE and LeCaRDv2 datasets.

Hugging Face dataset page: [RULER-dataset/RULER](https://huggingface.co/datasets/RULER-dataset/RULER)

## Availability

| Component | Status |
|---|---|
| README | Available |
| Framework figure | Available |
| Dataset page | Available on Hugging Face |
| Training code | Available |
| Evaluation scripts | Available |
| Checkpoints | Coming soon, subject to release approval |
| Reproduction guide | Available |

## Release Roadmap

- [x] Paper accepted to SIGIR 2026.
- [x] Initial README draft.
- [x] Clean training and evaluation scripts.
- [x] Organize data preprocessing pipeline.
- [x] Add reproducible shell entry points.
- [x] Release processed dataset instructions or links.
- [ ] Release trained checkpoints when permitted.
- [x] Add reproduction documentation.

## Repository Structure

The release focuses on the paper's main pipeline:

```text
RULER/
|-- retriever/                 # Dense retrieval and shared Qwen3 model
|-- scripts/                   # Training, data construction, and evaluation entrypoints
|-- docs/                      # Dataset and result notes
|-- tests/                     # Behavioral and checkpoint smoke tests
|-- build_train_dataset.py     # Stage 2 group construction
|-- build_test_dataset.py      # Evaluation candidate construction
|-- requirements.txt           # Python dependencies
`-- README.md
```

## Core Entry Points

| Task | Command |
|---|---|
| Train Stage 1 retriever | `bash scripts/train_ruler_retriever.sh` |
| Build retrieval rankings and Stage 2 groups | `bash scripts/run_ruler_data_pipeline.sh` |
| Train Stage 2 reranker | `bash scripts/train_ruler_reranker.sh` |
| Evaluate retrieval and reranking | `bash scripts/benchmark_ruler.sh` |

Both stages share the implementation in
`retriever/llm2vec_lasttoken/modeling_qwen3_embed.py`. The historical
`reranker/src/qwen3forall.py` module is retained as a compatibility import.

## Installation

RULER was recovered and verified with Python 3.9, PyTorch 2.4.0, CUDA 12.1,
and Transformers 4.52.4.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See [`docs/data.md`](docs/data.md) for the expected data layout. Define
`BASE_MODEL_DIR`, `DATA_DIR`, and `OUTPUT_DIR` before running the scripts.

## License

RULER source code is released under the [MIT License](LICENSE). Third-party
models, datasets, tokenizers, and separately distributed checkpoints remain
subject to their original licenses; see [`docs/licenses.md`](docs/licenses.md).

## Acknowledgements

RULER builds on open research resources including Qwen, JuDGE, LeCaRDv2, BGE, and prior work on unified retrieval and reranking. We thank the authors and maintainers of these resources for making legal information retrieval research more reproducible.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{hou2026ruler,
  title = {{RULER}: Robust Unified {LLM}-based Efficient Retrieval for Legal Information},
  author = {Chenyu Hou and Ziyang Wang and Bin Cao and Jiaxing Wang and Tianming Zhang and Tiantian Li},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2026},
  doi = {10.1145/3805712.3809698}
}
```

## Contact

For questions about the paper or release plan, please contact the authors through the paper correspondence information.
