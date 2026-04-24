# RULER
Code &amp; Datasets for RULER: Robust Unified LLM-based Efficient Retrieval for Legal Information

# RULER: Robust Unified LLM-based Efficient Retrieval for Legal Information

<p align="center">

  <a href="https://sigir.org/sigir2026/"><img src="https://img.shields.io/badge/Venue-SIGIR%202026-blue" alt="SIGIR 2026"></a>
  <a href="https://arxiv.org/abs/2501.00001"><img src="https://img.shields.io/badge/arXiv-2501.00001-green" alt="Paper arXiv"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow" alt="License: MIT"></a>

</p>

RULER is a unified legal retrieval framework that shares a single **Qwen3-0.6B** backbone across a bi-encoder retriever and a cross-encoder reranker. By introducing **distribution-robust training** with dynamic margin ranking loss and entropy-based regularization, RULER effectively suppresses phantom hits in zero-recall scenarios and achieves state-of-the-art performance on Chinese legal retrieval benchmarks.

> [!NOTE]
> This is the official code release for the paper accepted at **SIGIR 2026**. The datasets (JuDGE-Stat, LeCaRDv2-Stat) are distributed separately on Hugging Face. Full model weights will be released upon paper acceptance.

## Table of Contents

- [Key Contributions](#key-contributions)
- [Method](#method)
- [Main Results](#main-results)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Detailed Pipeline](#detailed-pipeline)
- [Model Zoo](#model-zoo)
- [Citation](#citation)
- [License](#license)

## Key Contributions

| | |
|---|---|
| **Unified LLM Backbone** | A single Qwen3-0.6B model serves both bi-encoder retrieval and cross-encoder reranking, reducing total parameters compared to using separate models. |
| **Distribution-Robust Reranking** | Novel combination of dynamic margin ranking loss and entropy regularization on all-negative groups, effectively reducing phantom hits (false positives at rank 1). |
| **Hard Negative Mining** | Top-k retrieval candidates are leveraged to construct mixed positive/negative groups and all-negative groups, enabling the reranker to handle diverse difficulty levels. |
| **Memory-Efficient Deployment** | Hot-Swap and Merge inference modes: loading only one model at a time reduces peak GPU memory by **65.3%** compared to a two-model pipeline. |

## Method

RULER follows a two-stage retrieve-then-rerank pipeline built on Qwen3:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RULER: Two-Stage Pipeline                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Query ──► [Qwen3] ──► Embedding ──► FAISS Search ──► Top-50 Candidates │
│            (Stage 1: Bi-Encoder, last-token pooling, margin loss)      │
│                                                   │                     │
│                                                   ▼                     │
│                                    Mixed Groups + All-Neg Groups        │
│                                                   │                     │
│                                                   ▼                     │
│  Candidates ──► [Qwen3 + LoRA] ──► Relevance Score ──► Re-ranked List  │
│                (Stage 2: Cross-Encoder, dynamic margin, entropy reg.)  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Stage 1 — Bi-Encoder Retrieval.** Fine-tune a Qwen3-based bi-encoder with last-token pooling and normalized embeddings for dense retrieval. A margin-based contrastive loss contrasts positive law articles against random negatives sampled from the corpus.

**Stage 2 — Cross-Encoder Reranking.** Retrieve top-k candidates with Stage 1, mine hard negatives, and construct two types of training groups:

- **Mixed groups** (3 per query): contain at least 1 positive and 9 retrieval-hard negatives.
- **All-negative groups** (2 per query): contain 10 negatives only, training the reranker to recognize zero-recall scenarios.

Apply **dynamic margin ranking loss** with confidence scaling and **entropy-based regularization** on all-negative groups to calibrate score distributions and suppress phantom hits.

## Main Results

Evaluated on **JuDGE-Stat** and **LeCaRDv2-Stat** benchmarks (NVIDIA RTX A6000, 44.55 GB VRAM, PyTorch 2.5.1+cu124).

### End-to-End Pipeline Efficiency (Table 1)

| Model | Params | Retrieval (ms/q) | Rerank (ms/q) | E2E (ms/q) | QPS | Peak Mem (GB) |
|---|---|---|---|---|---|---|
| **RULER (Merge)** | 596M | 55.72 | 1025.47 | **1081.19** | **0.925** | 27.92 |
| **RULER (Hot-Swap)** | 596M | 121.65 | 2543.92 | 2665.57 | 0.375 | 28.00 |
| BGE Pipeline | 558M | 30.37 | 1162.63 | 1193.00 | 0.838 | 10.94 |
| RoBERTa Pipeline | 102M | 18.55 | 307.76 | 326.31 | 3.065 | 4.77 |

### Retrieval Stage Efficiency (Table 2)

| Model | Params | ms/query | Query Enc (ms) | Doc Enc (ms/q) | Search (ms) | Peak Mem (GB) |
|---|---|---|---|---|---|---|
| **RULER** | 596M | 46.82 | 30.56 | 12.99 | 0.061 | 5.67 |
| BGE-M3 | 567M | 30.77 | 17.48 | 7.05 | 0.068 | 3.69 |
| RoBERTa | 102M | 15.76 | 11.11 | 2.57 | 0.059 | 1.58 |

### Reranking Throughput (Table 3)

| Model | Params | ms/pair (b=256) | pairs/s (b=256) | Peak Mem (GB) |
|---|---|---|---|---|
| **RULER (Merge)** | 596M | **20.59** | **48.57** | 27.92 |
| **RULER (Hot-Swap)** | 596M | 26.73 | 37.41 | 28.00 |
| BGE-Reranker | 558M | 18.73 | 53.38 | 7.88 |
| RoBERTa | 102M | 6.14 | 162.81 | 4.77 |

### Memory Efficiency — Hot-Swap vs. Two-Model (Table 4)

| Strategy | Peak Mem (GB) | Load Time (s) | Memory Saved |
|---|---|---|---|
| **Hot-Swap** | **2.37** | 1.51 | **65.3%** |
| Two-Model | 6.83 | 2.71 | — |

### Precision — BF16 vs. FP32 (Table 5)

| Strategy | Batch Size | pairs/s | ms/pair | Peak Mem (GB) | Speedup vs FP32 |
|---|---|---|---|---|---|
| **Merge BF16** | 256 | **48.6** | **20.59** | 27.92 | **2.76×** |
| Hot-Swap BF16 | 256 | 37.4 | 26.73 | 28.00 | 3.38× |
| Merge FP32 | 64 | 17.6 | 56.77 | 8.93 | — |
| Hot-Swap FP32 | 64 | 11.1 | 90.23 | 9.01 | — |

## Dataset

The datasets (JuDGE-Stat, LeCaRDv2-Stat) are distributed separately on Hugging Face:

```
https://huggingface.co/datasets/<org>/<dataset>
```

After downloading, place the files under `data/` with the following layout:

```
data/
|-- train.json
|-- test.json
|-- law_corpus.jsonl
|-- case_corpus.jsonl      # optional
|-- qrels_file_train
`-- qrels_file_test
```

See [docs/data.md](docs/data.md) for the full dataset documentation.

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- CUDA >= 11.8 (GPU recommended)
- NVIDIA GPU with at least 16 GB VRAM (32 GB recommended for full pipeline)

### Install dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >= 2.0.0 | Deep learning framework |
| `transformers` | >= 4.40.0 | Model architecture |
| `accelerate` | latest | Distributed training |
| `peft` | latest | LoRA fine-tuning |
| `faiss-cpu` / `faiss-gpu` | latest | Dense retrieval index |
| `numpy`, `pandas`, `tqdm`, `scikit-learn` | latest | Data processing & evaluation |
| `jieba` | 0.42.1 | Chinese text segmentation |

### Download base model

```bash
huggingface-cli download Qwen/Qwen3-0.6B --repo-type model --local-dir /path/to/qwen3_base
```

### Download dataset

```bash
huggingface-cli download <org>/<dataset> --repo-type dataset --local-dir data/
```

## Quick Start

### Full pipeline

```bash
# 1. Set environment variables
export DATA_DIR=/path/to/data
export BASE_MODEL_DIR=/path/to/qwen3_base
export OUTPUT_DIR=/path/to/outputs

# 2. Train Stage 1 retriever
bash scripts/train_ruler_retriever.sh

# 3. Build ranking files and grouped TSVs
bash scripts/run_ruler_data_pipeline.sh

# 4. Train Stage 2 reranker
bash scripts/train_ruler_reranker.sh

# 5. Evaluate
bash scripts/benchmark_ruler.sh
```

### Individual stages

```bash
# Train Stage 1 retriever
DATA_DIR=/path/to/data \
BASE_MODEL_DIR=/path/to/qwen3_base \
OUTPUT_DIR=/path/to/outputs \
bash scripts/train_ruler_retriever.sh

# Build top-k candidate rankings
QUERY_FILE=/path/to/train.json \
CORPUS_FILE=/path/to/law_corpus.jsonl \
RETRIEVER_MODEL_PATH=/path/to/stage1_retriever \
BASE_MODEL_PATH=/path/to/qwen3_base \
OUTPUT_RANK_PATH=/path/to/rank_train.tsv \
bash scripts/build_ruler_rankings.sh

# Build grouped TSV for reranker training
TRAIN_JSON=/path/to/train.json \
LAW_CORPUS=/path/to/law_corpus.jsonl \
RANK_TRAIN=/path/to/rank_train.tsv \
OUT_FILE=/path/to/train_grouped.tsv \
bash scripts/build_ruler_train_tsv.sh

# Train Stage 2 reranker
DATA_DIR=/path/to/data \
OUTPUT_DIR=/path/to/outputs \
bash scripts/train_ruler_reranker.sh

# Evaluate retrieval, reranking, robustness, and efficiency
bash scripts/benchmark_ruler.sh
```

See [docs/reproduction.md](docs/reproduction.md) for the complete reproduction guide.

## Repository Layout

```
.
|-- retriever/                          # Core retrieval & reranking implementation
|   |-- llm2vec_lasttoken/              # Qwen3-based bi-encoder & cross-encoder
|   |   |-- train_qwen3_full.py         # Stage 1: retriever training
|   |   |-- modeling_qwen3_embed.py      # Model architecture (last-token pooling)
|   |   |-- encode_with_qwen3.py        # Encoding utilities
|   |   |-- reranker/
|   |       |-- src/
|   |           |-- train_lora.py       # Stage 2: LoRA reranker training
|   |           |-- eval_deep_ours.py    # Reranking evaluation
|   |           |-- calc_dist_metrics.py # Robustness metrics
|   |-- benchmark/                      # Benchmark scripts & output
|   |   |-- benchmark_ruler_rerank.py   # End-to-end inference benchmark
|   |   |-- benchmark_retrieval.py      # Retrieval-only benchmark
|   |   |-- benchmark_e2e_pipeline.py  # Full pipeline benchmark
|   |-- dense/                           # FAISS-based dense retrieval
|-- baselines/                           # Baseline implementations
|   |-- bm25/                           # BM25 sparse retrieval
|   |-- bge/                            # BGE Reranker
|   |-- bgem3/                          # BGE-M3 dense retrieval
|   |-- ur2n/                           # UR2N unified retriever
|-- scripts/                             # Portable release-facing wrappers
|   |-- train_ruler_retriever.sh        # Stage 1 training
|   |-- train_ruler_reranker.sh         # Stage 2 training
|   |-- benchmark_ruler.sh              # Evaluation
|   |-- build_ruler_rankings.sh         # Top-k candidate ranking
|   |-- build_ruler_train_tsv.sh        # Training group TSV
|   |-- build_ruler_test_tsv.sh         # Test group TSV
|   |-- filter_lecard_subset.sh         # LeCaRD subset filtering
|   |-- run_ruler_data_pipeline.sh      # Full pipeline orchestrator
|-- docs/                                # Documentation
|   |-- data.md                         # Dataset guide
|   |-- reproduction.md                 # Reproduction guide
|   |-- results.md                      # Results and evaluation notes
|   |-- paper_code_map.md               # Paper-to-code mapping
|   |-- data_pipeline.md                # Pipeline architecture
|-- ruler_release/                       # Release-specific assets
|   |-- README.md
|   `-- LICENSE
|-- build_train_dataset.py               # Training group TSV builder
|-- build_test_dataset.py                # Test group TSV builder
|-- build_dataset.py                     # General TSV builder
|-- check_data.py                        # LeCaRD subset filter
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Detailed Pipeline

### Data Preparation

```
Raw Cases (JSON) → split.py → train.json, test.json
                              → gen_qrels_file.py → qrels_file_train, qrels_file_test
                              → gen_dense_train.py → dense_train.json (Stage 1)
                              → law_corpus.jsonl (corpus for retrieval)
```

### Stage 1 — Retriever Training

```
dense_train.json + Qwen3-0.6B → train_qwen3_full.py → Stage 1 Checkpoint
                                              ↓
                        Qwen3-Bi-Encoder (last-token pool, margin loss)
```

Key hyperparameters: `lr=1e-5`, `batch=4`, `query_len=512`, `doc_len=200`, `epochs=5`, `margin=0.2`.

### Stage 2 — Reranker Training

```
Stage 1 Checkpoint → build_ruler_rankings.sh → rank_*.tsv (top-k candidates)
                                          → build_ruler_train_tsv.sh → train_grouped.tsv
                                                                              ↓
Stage 1 Checkpoint + train_grouped.tsv → train_lora.py → RULER Reranker
                                                  (LoRA, dynamic margin,
                                                   entropy regularization)
```

Key hyperparameters: `LoRA r=32`, `alpha=64`, `lr=1e-5`, `epochs=5`, `margin=1.0`, `alpha=3`, `gamma=0.05`, `beta_ent=0.2`, `beta_IRDA=0.1`.

See [docs/reproduction.md](docs/reproduction.md) for the complete pipeline.

## Model Zoo

| Model | Base | Stage | Description |
|-------|------|-------|-------------|
| RULER-Retriever | Qwen3-0.6B | Stage 1 | Bi-encoder with last-token pooling |
| RULER-Reranker | Qwen3-0.6B + LoRA | Stage 2 | Cross-encoder with distribution-robust training |

Model weights will be released on Hugging Face upon paper acceptance.

## Citation

```bibtex
@inproceedings{hou2026ruler,
  title={{RULER}: Robust Unified {LLM}-based Efficient Retrieval for Legal Information},
  author={Chenyu Hou and Ziyang Wang and Bin Cao and Jiaxing Wang and Tianming Zhang and Tiantian Li},
  booktitle={SIGIR 2026},
  year={2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE). Third-party base models and tokenizers remain subject to their own licenses. See [docs/licenses.md](docs/licenses.md) for details.

## Dataset
Due to GitHub's file size limitations, the pre-processed datasets used for training and evaluation are hosted on Hugging Face:
🤗 [Hugging Face Dataset](https://huggingface.co/datasets/RULER-dataset/RULER)
