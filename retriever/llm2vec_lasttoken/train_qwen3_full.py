import os
import json
import torch
import random
import logging
import argparse
from dataclasses import dataclass
from typing import List, Dict

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
    PreTrainedTokenizerBase
)

from modeling_qwen3_embed import Qwen3ForEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def enable_torch24_dtensor_compat():
    """Expose Torch 2.4's DTensor at the location expected by Transformers 4.52."""
    try:
        from torch.distributed.tensor import DTensor  # noqa: F401
    except ImportError:
        from torch.distributed._tensor import DTensor
        import torch.distributed.tensor as tensor_module
        import transformers.modeling_utils as modeling_utils

        tensor_module.DTensor = DTensor
        modeling_utils.DTensor = DTensor


@dataclass
class Args:
    model_name_or_path: str
    tokenizer_name: str
    train_path: str
    output_dir: str
    q_max_len: int = 512
    p_max_len: int = 200
    per_device_train_batch_size: int = 4
    learning_rate: float = 5e-5
    num_train_epochs: int = 5
    max_steps: int = -1
    save_strategy: str = "epoch"
    logging_steps: int = 50
    seed: int = 42


@dataclass
class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line.strip()) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "query": item["query"],
            "pos": random.choice(item["positives"]),
            "neg": random.choice(item["negatives"]),
        }


@dataclass
class Collator:
    tokenizer: PreTrainedTokenizerBase
    max_q_len: int
    max_p_len: int

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        q_texts = [item["query"] for item in batch]
        p_texts = [item["pos"] for item in batch]
        n_texts = [item["neg"] for item in batch]

        q = self.tokenizer(q_texts, padding=True, truncation=True, max_length=self.max_q_len, return_tensors="pt")
        p = self.tokenizer(p_texts, padding=True, truncation=True, max_length=self.max_p_len, return_tensors="pt")
        n = self.tokenizer(n_texts, padding=True, truncation=True, max_length=self.max_p_len, return_tensors="pt")

        return {"q_input": q, "p_input": p, "n_input": n}


def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), sequence_lengths]


class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query_input_ids = inputs["q_input"]["input_ids"].to(model.device)
        query_attention_mask = inputs["q_input"]["attention_mask"].to(model.device)
        pos_input_ids = inputs["p_input"]["input_ids"].to(model.device)
        pos_attention_mask = inputs["p_input"]["attention_mask"].to(model.device)
        neg_input_ids = inputs["n_input"]["input_ids"].to(model.device)
        neg_attention_mask = inputs["n_input"]["attention_mask"].to(model.device)

        query_outputs = model(input_ids=query_input_ids, attention_mask=query_attention_mask, mode="embedding")
        pos_outputs = model(input_ids=pos_input_ids, attention_mask=pos_attention_mask, mode="embedding")
        neg_outputs = model(input_ids=neg_input_ids, attention_mask=neg_attention_mask, mode="embedding")

        query_embeds = query_outputs["embeddings"]
        pos_embeds = pos_outputs["embeddings"]
        neg_embeds = neg_outputs["embeddings"]

        margin = 0.2
        pos_scores = (query_embeds * pos_embeds).sum(dim=-1)
        neg_scores = (query_embeds * neg_embeds).sum(dim=-1)
        loss = torch.nn.functional.relu(margin - pos_scores + neg_scores).mean()

        return (loss, None) if return_outputs else loss


def main():
    enable_torch24_dtensor_compat()
    parser = HfArgumentParser(Args)
    if len(os.sys.argv) == 2 and os.sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=os.path.abspath(os.sys.argv[1]))[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)


    model = Qwen3ForEmbedding.from_pretrained(args.model_name_or_path)




    model.supports_gradient_checkpointing = True


    model.config.use_cache = False



    dataset = SupervisedDataset(args.train_path)
    collator = Collator(tokenizer, args.q_max_len, args.p_max_len)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        report_to="none",
        fp16=False,
        bf16=True,
        ddp_find_unused_parameters=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_checkpointing=True
    )

    trainer = ContrastiveTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print("Training completed; model saved to", args.output_dir)


if __name__ == "__main__":
    main()
