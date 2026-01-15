"""Mean Difference Vector Computation."""
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


def compute_mean_diff(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    harmful_file: str,
    harmless_file: str,
    layer: int,
    output_dir: str,
    model_type: str = "llama2",
    limit: Optional[int] = None,
    normalize: bool = True,
) -> Tensor:
    """Compute mean difference vector: mean(harmful) - mean(harmless)."""
    harmful = load_prompts(harmful_file, limit)
    harmless = load_prompts(harmless_file, limit)
    
    logger.info(f"Extracting {len(harmful)} harmful prompts")
    H_harm = extract_activations(model, tokenizer, harmful, layer, model_type)
    
    logger.info(f"Extracting {len(harmless)} harmless prompts")
    H_safe = extract_activations(model, tokenizer, harmless, layer, model_type)
    
    diff = H_harm.mean(dim=0) - H_safe.mean(dim=0)
    if normalize:
        diff = diff / (torch.norm(diff) + 1e-8)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(diff, f"{output_dir}/mean_harm.pt")
    logger.info(f"Saved mean_harm.pt")
    
    return diff
