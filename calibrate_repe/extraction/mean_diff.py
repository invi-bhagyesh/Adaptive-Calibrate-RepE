"""Mean-difference vector computation.

Used for both the baseline harmfulness direction (v_harm via mean diff)
and the refusal vector (v_refuse = mean(refused) - mean(accepted)).
"""
import logging
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data import load_prompts
from .activations import extract_activations

logger = logging.getLogger(__name__)


def compute_mean_diff(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    positive_file: str,
    negative_file: str,
    layer: int,
    output_dir: str,
    output_name: str = "mean_diff.pt",
    model_type: str = "llama2",
    limit: Optional[int] = None,
    normalize: bool = True,
) -> Tensor:
    """Compute ``mean(positive) - mean(negative)`` at *layer* and save.

    For a harmfulness direction::

        compute_mean_diff(..., positive_file="harmful.json",
                          negative_file="harmless.json",
                          output_name="mean_harm.pt")

    For a refusal vector::

        compute_mean_diff(..., positive_file="refused.json",
                          negative_file="accepted.json",
                          output_name="v_refuse.pt")
    """
    pos_prompts = load_prompts(positive_file, limit)
    neg_prompts = load_prompts(negative_file, limit)

    logger.info(f"Extracting {len(pos_prompts)} positive prompts")
    H_pos = extract_activations(model, tokenizer, pos_prompts, layer, model_type)

    logger.info(f"Extracting {len(neg_prompts)} negative prompts")
    H_neg = extract_activations(model, tokenizer, neg_prompts, layer, model_type)

    diff = H_pos.mean(dim=0) - H_neg.mean(dim=0)
    if normalize:
        diff = diff / (torch.norm(diff) + 1e-8)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(diff, f"{output_dir}/{output_name}")
    logger.info(f"Saved {output_name}")

    return diff
