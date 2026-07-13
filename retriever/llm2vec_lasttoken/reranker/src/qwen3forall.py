"""Compatibility imports for the shared RULER model implementation."""

import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from modeling_qwen3_embed import Qwen3ForEmbedding, last_token_pool  # noqa: E402

__all__ = ["Qwen3ForEmbedding", "last_token_pool"]
