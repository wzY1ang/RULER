"""Load a real RULER checkpoint and run embedding/reranking forward passes."""

import argparse
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer


MODEL_DIR = Path(__file__).resolve().parents[1] / "retriever" / "llm2vec_lasttoken"
sys.path.insert(0, str(MODEL_DIR))

from modeling_qwen3_embed import Qwen3ForEmbedding  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = Qwen3ForEmbedding.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(args.device)
    model.config.bidirectional = True
    model.eval()

    inputs = tokenizer(
        ["contract dispute", "criminal law article"],
        padding=True,
        return_tensors="pt",
    ).to(args.device)
    with torch.no_grad():
        embeddings = model(**inputs, mode="embedding")["embeddings"]
        scores = model(**inputs, mode="rerank")["scores"]

    if embeddings.shape[0] != 2 or scores.shape != (2,):
        raise RuntimeError(f"Unexpected shapes: embeddings={embeddings.shape}, scores={scores.shape}")
    if not torch.isfinite(embeddings).all() or not torch.isfinite(scores).all():
        raise RuntimeError("Checkpoint produced non-finite outputs")

    print(f"checkpoint={args.checkpoint}")
    print(f"device={args.device} dtype={dtype}")
    print(f"embeddings={tuple(embeddings.shape)} scores={tuple(scores.shape)}")


if __name__ == "__main__":
    main()
