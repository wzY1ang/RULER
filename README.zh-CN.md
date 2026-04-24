# RULER

**RULER: 面向法律信息检索的鲁棒统一式 LLM 高效检索框架**

<p align="center">
  <a href="https://sigir.org/sigir2026/"><img src="https://img.shields.io/badge/Venue-SIGIR%202026-blue" alt="SIGIR 2026"></a>
  <img src="https://img.shields.io/badge/Status-Pre--release-orange" alt="Pre-release">
  <img src="https://img.shields.io/badge/Code-Coming%20Soon-lightgrey" alt="Code coming soon">
</p>

**语言：** [English](README.md) | 简体中文

本仓库是 **RULER** 的官方仓库，对应论文已被 **SIGIR 2026** 接收。

> **当前状态。** 论文已经录用，但公开代码仍在整理中。本仓库目前处于预发布阶段，代码、脚本、模型权重和完整复现实验说明会在清理与验证完成后逐步发布。

## 目录

- [动态](#动态)
- [概述](#概述)
- [框架图](#框架图)
- [为什么选择 RULER](#为什么选择-ruler)
- [方法](#方法)
- [实验结果](#实验结果)
- [数据集](#数据集)
- [发布计划](#发布计划)
- [计划中的仓库结构](#计划中的仓库结构)
- [引用](#引用)
- [联系](#联系)

## 动态

- **2026-07：** RULER 将发表于澳大利亚墨尔本举办的 SIGIR 2026。
- **2026：** 论文已被第 49 届 ACM SIGIR Conference on Research and Development in Information Retrieval 接收。
- **预发布：** 仓库清理和公开发布准备正在进行中。

## 概述

法律信息检索同时要求高召回率和高精度。传统的 retrieve-then-rerank 系统通常使用两个独立模型：一个双塔检索器负责候选生成，一个交叉编码器重排器负责精细排序。这种分离式设计虽然有效，但会带来参数冗余、部署复杂度以及两个阶段之间的语义不一致问题。

**RULER** 使用统一架构解决上述问题。该框架基于单个 **Qwen3-0.6B** 主干模型，先将其适配为稠密检索器，再进一步通过 LoRA 适配为交叉编码器重排器，使检索和重排两个阶段共享同一套表示能力，同时保留两阶段检索流程的效率优势。

RULER 尤其关注法律检索中的高置信错误候选问题。我们将这类被错误排到高位的无关候选称为 **Phantom Hits**，并在重排训练阶段显式建模这种场景。

## 框架图

<p align="center">
  <img src="assets/ruler-framework.png" alt="RULER 框架图" width="100%">
</p>

RULER 由三个连续部分组成：Stage 1 检索微调、基于检索结果的数据构造，以及 Stage 2 分布感知重排。训练过程同时显式建模混合正负样本组和全负零召回样本组。

## 为什么选择 RULER

- **统一检索与重排：** 使用同一个 LLM 主干同时支持候选生成和重排。
- **渐进式训练：** 重排器从检索阶段适配后的 checkpoint 初始化，而不是从无关模型重新开始。
- **鲁棒负例建模：** 同时构造混合正负样本组和全负样本组，模拟正常排序与零召回场景。
- **分布鲁棒目标：** 动态间隔排序目标和最大熵正则化用于降低模型对无关候选的过度自信。
- **面向法律领域评估：** 实验覆盖 JuDGE-Stat 和 LeCaRDv2-Stat 两个中文法律法条检索基准。

## 方法

RULER 保留经典的 retrieve-then-rerank 流程，但在两个阶段之间共享底层主干模型：

```text
查询
  -> Stage 1: Qwen3 双塔检索器
  -> FAISS 稠密向量检索
  -> Top-50 候选法条
  -> Stage 2: Qwen3 + LoRA 交叉编码器重排器
  -> 最终排序结果
```

### Stage 1：双塔检索

检索阶段将 Qwen3-0.6B 微调为双塔检索器，使用 last-token pooling 和归一化向量表示查询与法条，并通过 FAISS 检索生成候选集合。

### Stage 2：交叉编码器重排

重排阶段从 Stage 1 checkpoint 初始化，并使用 LoRA 进行参数高效微调。训练样本由检索候选构造，包含两类组：

- **混合样本组：** 包含相关法条和检索得到的困难负例法条。
- **全负样本组：** 仅包含无关候选，用于模拟零召回检索输出。

RULER 将动态间隔排序损失与最大熵正则化结合，在混合样本组中强化正负区分，在全负样本组中保持校准后的不确定性。

## 实验结果

完整实验设置会随清理后的代码和复现脚本一同发布。以下结果来自已录用论文。

### 检索结果

| 数据集 | MRR@100 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| JuDGE-Stat | 0.8965 | 0.6570 | 0.8150 |
| LeCaRDv2-Stat | 0.9001 | 0.6832 | 0.8350 |

### 重排结果

| 数据集 | NDCG@10 | MAP@10 | R-Prec | MRR@10 |
|---|---:|---:|---:|---:|
| JuDGE-Stat | 0.9013 | 0.8299 | 0.7672 | 0.9693 |
| LeCaRDv2-Stat | 0.8605 | 0.7653 | 0.7157 | 0.9498 |

### 对 Phantom Hits 的鲁棒性

在 JuDGE-Stat 上，RULER 相比代表性统一式基线显著降低了高置信噪声排序风险：

| 方法 | NR@R ↓ | Overlap ↓ | NDCG@10 ↑ |
|---|---:|---:|---:|
| UR2N-RandengT5 | 50.6% | 89.6% | 0.7888 |
| BGE-M3 (Finetuned) | 12.7% | 79.2% | 0.8462 |
| RULER | 9.9% | 66.7% | 0.9013 |

## 数据集

RULER 在两个中文法律法条检索基准上进行评估：

| 数据集 | 来源 | 查询数 | 说明 |
|---|---|---:|---|
| JuDGE-Stat | JuDGE | 2,505 | 小样本法律法条检索基准。 |
| LeCaRDv2-Stat | LeCaRDv2 | 39,833 | 从 LeCaRDv2 构建的精炼大规模子集，用于降低标注稀疏性并提升结构一致性。 |

数据集页面已在 Hugging Face 上提供。数据预处理脚本和详细使用说明会在代码清理与许可检查完成后发布。用户也应遵守原始 JuDGE 和 LeCaRDv2 数据集的使用条款。

Hugging Face 数据集页面：[RULER-dataset/RULER](https://huggingface.co/datasets/RULER-dataset/RULER)

## 发布计划

- [x] 论文被 SIGIR 2026 接收。
- [x] 初版 README。
- [ ] 清理训练和评估脚本。
- [ ] 整理数据预处理流程。
- [ ] 添加可复现实验配置。
- [ ] 发布处理后数据说明或链接。
- [ ] 在许可允许时发布训练好的 checkpoint。
- [ ] 添加完整复现文档。

## 计划中的仓库结构

最终发布预计采用类似结构：

```text
RULER/
|-- retriever/                 # Stage 1 稠密检索器
|-- reranker/                  # Stage 2 LoRA 重排器
|-- baselines/                 # 稀疏、稠密和统一式基线
|-- scripts/                   # 训练、数据构建和评估入口
|-- configs/                   # 可复现实验配置
|-- docs/                      # 数据集、复现和实现说明
|-- requirements.txt           # Python 依赖
|-- LICENSE
`-- README.md
```

该结构可能会在代码清理过程中略有调整。

## 安装

安装命令会随首个代码版本一起发布。预计环境如下：

- Python 3.9+
- PyTorch 2.0+
- 用于训练和评估的 CUDA GPU
- `transformers`、`accelerate`、`peft`、`faiss`、`numpy`、`pandas` 以及常用评估工具

## 引用

如果本工作对你有帮助，请引用：

```bibtex
@inproceedings{hou2026ruler,
  title = {{RULER}: Robust Unified {LLM}-based Efficient Retrieval for Legal Information},
  author = {Chenyu Hou and Ziyang Wang and Bin Cao and Jiaxing Wang and Tianming Zhang and Tiantian Li},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2026}
}
```

## 联系

如有关于论文或发布计划的问题，请通过论文中的通讯信息联系作者。
