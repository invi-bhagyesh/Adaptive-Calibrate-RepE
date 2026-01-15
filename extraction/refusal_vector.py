"""Refusal Vector Extraction - Mean diff between refused/accepted responses."""
import logging
from pathlib import Path
from typing import List, Optional

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_prompts
from .lat_probe import extract_activations

logger = logging.getLogger(__name__)


def compute_refusal_vector(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    refused_file: str,
    accepted_file: str,
    layer: int,
    output_dir: str,
    model_type: str = "llama2",
    limit: Optional[int] = None,
) -> Tensor:
    """Compute refusal vector: mean(refused) - mean(accepted) at t_post-inst."""
    refused = load_prompts(refused_file, limit)
    accepted = load_prompts(accepted_file, limit)
    
    logger.info(f"Extracting {len(refused)} refused prompts")
    H_ref = extract_activations(model, tokenizer, refused, layer, model_type)
    
    logger.info(f"Extracting {len(accepted)} accepted prompts")
    H_acc = extract_activations(model, tokenizer, accepted, layer, model_type)
    
    diff = H_ref.mean(dim=0) - H_acc.mean(dim=0)
    diff = diff / (torch.norm(diff) + 1e-8)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(diff, f"{output_dir}/v_refuse.pt")
    logger.info(f"Saved v_refuse.pt")
    
    return diff
