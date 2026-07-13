"""Search encoded passages with an exact FAISS inner-product index."""

import argparse
import glob

import faiss
import numpy as np
import torch


def load_encoded(path_pattern):
    files = sorted(glob.glob(path_pattern))
    if not files:
        raise FileNotFoundError(f"No encoded files matched: {path_pattern}")

    tensors = []
    identifiers = []
    for path in files:
        embeddings, ids = torch.load(path, map_location="cpu", weights_only=True)
        tensors.append(embeddings.float())
        identifiers.extend(ids)
    return torch.cat(tensors).numpy(), identifiers


def search(index, queries, depth, batch_size):
    if batch_size <= 0:
        return index.search(queries, depth)

    score_batches = []
    index_batches = []
    for start in range(0, len(queries), batch_size):
        scores, indices = index.search(queries[start:start + batch_size], depth)
        score_batches.append(scores)
        index_batches.append(indices)
    return np.concatenate(score_batches), np.concatenate(index_batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_reps", required=True)
    parser.add_argument("--passage_reps", required=True)
    parser.add_argument("--save_ranking_to", required=True)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    passage_reps, passage_ids = load_encoded(args.passage_reps)
    query_reps, query_ids = load_encoded(args.query_reps)
    depth = min(args.depth, len(passage_ids))

    index = faiss.IndexFlatIP(passage_reps.shape[1])
    index.add(passage_reps)
    scores, indices = search(index, query_reps, depth, args.batch_size)

    with open(args.save_ranking_to, "w", encoding="utf-8") as handle:
        for query_id, query_scores, query_indices in zip(query_ids, scores, indices):
            for score, passage_index in zip(query_scores, query_indices):
                handle.write(f"{query_id}\t{passage_ids[passage_index]}\t{score}\n")


if __name__ == "__main__":
    main()
