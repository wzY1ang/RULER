import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "retriever" / "llm2vec_lasttoken"
sys.path.insert(0, str(MODEL_DIR))

from modeling_qwen3_embed import Qwen3ForEmbedding  # noqa: E402

RERANKER_DIR = MODEL_DIR / "reranker" / "src"
sys.path.insert(0, str(RERANKER_DIR))
from train_lora import groupwise_loss  # noqa: E402
from eval_deep_ours import calc_global_detailed_metrics  # noqa: E402


class ModelForwardTest(unittest.TestCase):
    def _make_model(self):
        torch.manual_seed(7)
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
            attention_dropout=0.0,
            use_cache=False,
        )
        config.parallel_attn = False
        config.sliding_window = None
        config.base_model_tp_plan = {}
        config._attn_implementation = "eager"
        return Qwen3ForEmbedding(config).eval()

    def test_embedding_and_reranking_outputs(self):
        model = self._make_model()
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mask = torch.ones_like(input_ids)

        embeddings = model(input_ids, attention_mask=mask, mode="embedding")["embeddings"]
        scores = model(input_ids, attention_mask=mask, mode="rerank")["scores"]

        self.assertEqual(embeddings.shape, (2, model.config.hidden_size))
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-6))

class GroupBuilderTest(unittest.TestCase):
    def test_paper_group_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            corpus = temp / "corpus.jsonl"
            queries = temp / "train.json"
            ranking = temp / "rank.tsv"
            output = temp / "groups.tsv"

            corpus.write_text(
                "".join(
                    json.dumps({"text_id": str(i), "text": f"law {i}"}) + "\n"
                    for i in range(1, 31)
                ),
                encoding="utf-8",
            )
            queries.write_text(
                json.dumps([{"text_id": "q1", "text": "query", "la": ["1"]}]),
                encoding="utf-8",
            )
            ranking.write_text(
                "".join(f"q1\t{i}\t{1.0 / i}\n" for i in range(1, 31)),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "build_train_dataset.py"),
                    "--train_json",
                    str(queries),
                    "--law_corpus",
                    str(corpus),
                    "--rank_train",
                    str(ranking),
                    "--out_file",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

            groups = {}
            for row in rows:
                groups.setdefault(row["group_id"], []).append(row)

            self.assertEqual(len(rows), 50)
            self.assertEqual(len(groups), 5)
            self.assertEqual(sum("_mix_" in key for key in groups), 3)
            self.assertEqual(sum("_neg_" in key for key in groups), 2)
            self.assertTrue(
                all(any(row["cand_label"] == "1" for row in group) for key, group in groups.items() if "_mix_" in key)
            )
            self.assertTrue(
                all(all(row["cand_label"] == "0" for row in group) for key, group in groups.items() if "_neg_" in key)
            )


class FaissSearchTest(unittest.TestCase):
    def test_exact_inner_product_ranking(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            queries = temp / "queries.pt"
            passages = temp / "passages.pt"
            ranking = temp / "ranking.tsv"
            torch.save((torch.tensor([[1.0, 0.0]]), ["q1"]), queries)
            torch.save(
                (torch.tensor([[0.0, 1.0], [1.0, 0.0]]), ["p1", "p2"]),
                passages,
            )

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "retriever" / "search_faiss.py"),
                    "--query_reps",
                    str(queries),
                    "--passage_reps",
                    str(passages),
                    "--save_ranking_to",
                    str(ranking),
                    "--depth",
                    "2",
                ],
                check=True,
            )

            rows = ranking.read_text(encoding="utf-8").splitlines()
            self.assertTrue(rows[0].startswith("q1\tp2\t"))


class RerankerObjectiveTest(unittest.TestCase):
    def test_ruler_dynamic_margin_configuration(self):
        scores = torch.tensor([2.0, 0.0])
        labels = torch.tensor([1, 0])
        loss = groupwise_loss(
            scores,
            labels,
            [2],
            torch.tensor([1.0]),
            margin=1.0,
            alpha=3.0,
            gamma=0.0,
            beta_IRDA=0.0,
        )
        delta = scores[0] - scores[1]
        dynamic_margin = 1.0 + 3.0 * torch.sigmoid(-delta)
        expected = torch.nn.functional.softplus(-(delta - dynamic_margin))
        self.assertTrue(torch.allclose(loss, expected))

    def test_uniform_all_negative_group_has_zero_entropy_penalty(self):
        loss = groupwise_loss(
            torch.zeros(3),
            torch.zeros(3, dtype=torch.long),
            [3],
            torch.tensor([0.0]),
        )
        self.assertTrue(torch.allclose(loss, torch.tensor(0.0), atol=1e-7))

    def test_global_calibration_metrics(self):
        metrics = calc_global_detailed_metrics(
            [{"labels": [1, 0, 1, 0], "raw_scores": [3.0, -1.0, 2.0, 0.0]}]
        )
        self.assertAlmostEqual(metrics["Raw_Logits_Gap"], 3.0)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
