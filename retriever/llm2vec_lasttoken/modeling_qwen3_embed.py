from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model


def last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the final non-padding token from each sequence."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden[:, -1]
    idx = attention_mask.sum(dim=1) - 1
    b = last_hidden.shape[0]
    return last_hidden[torch.arange(b, device=last_hidden.device), idx]


class Qwen3ForEmbedding(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)

        if not hasattr(config, "parallel_attn") or config.parallel_attn is None:
            config.parallel_attn = False

        # RULER uses data parallelism. Disable the unused Transformers tensor
        # parallel plan to keep single-device and DDP checkpoint loading stable.
        config.base_model_tp_plan = {}

        self.model = Qwen3Model(config)
        self.score_head = nn.Linear(config.hidden_size, 1)

        nn.init.normal_(self.score_head.weight, mean=0.0, std=0.02)
        if self.score_head.bias is not None:
            nn.init.zeros_(self.score_head.bias)

        # The embedding and reranking paths do not use the KV cache.
        self.model.config.use_cache = False

        # Prefer PyTorch scaled dot-product attention when no backend is set.
        try:
            if not hasattr(self.model.config, "attn_implementation") or self.model.config.attn_implementation is None:
                self.model.config.attn_implementation = "sdpa"
        except Exception:
            pass

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mode: str = "embedding",
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)

        kwargs["output_hidden_states"] = False
        kwargs["return_dict"] = True
        kwargs["use_cache"] = False

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        if mode == "embedding":
            last_hidden = outputs.last_hidden_state
            pooled = last_token_pool(last_hidden, attention_mask)
            return {"embeddings": F.normalize(pooled, p=2, dim=1)}

        if mode == "rerank":
            last_hidden = outputs.last_hidden_state
            pooled = last_token_pool(last_hidden, attention_mask)

            scores = self.score_head(pooled).squeeze(-1)

            return {"scores": scores, "pooled": pooled}

        return outputs
