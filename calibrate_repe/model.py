"""Model loading and layer access utilities."""
import logging
from typing import List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config

logger = logging.getLogger(__name__)


def get_layers(model: nn.Module) -> List[nn.Module]:
    """Return the list of transformer layers from a HuggingFace model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)          # Llama, Qwen, Mistral
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)          # GPT-2 / GPT-Neo style
    raise ValueError("Unknown model architecture — cannot locate transformer layers")


def load_model(config: Config):
    """Load model and tokenizer from config."""
    logger.info(f"Loading model from {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer
