"""Shared activation extraction from transformer layers."""
import logging
from typing import List

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..model import get_layers
from ..data import format_prompt

logger = logging.getLogger(__name__)


def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    layer: int,
    model_type: str = "llama2",
    token_pos: int = -1,
) -> Tensor:
    """Extract hidden-state activations at *layer* and *token_pos* for each prompt.

    Returns a ``(len(prompts), hidden_dim)`` float tensor on CPU.
    """
    model.eval()
    layers = get_layers(model)
    activations: List[Tensor] = []
    storage: List[Tensor | None] = [None]

    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        storage[0] = h[:, token_pos, :].detach().cpu()

    handle = layers[layer].register_forward_hook(hook)
    try:
        for prompt in tqdm(prompts, desc=f"Extracting L{layer}"):
            formatted = format_prompt(prompt, model_type)
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs)
            activations.append(storage[0])
    finally:
        handle.remove()

    return torch.cat(activations, dim=0)
