"""LAT Probe - Linear readout for harmfulness at t_inst position."""
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import format_prompt, load_prompts

logger = logging.getLogger(__name__)


def get_layers(model: nn.Module) -> List[nn.Module]:
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return list(model.transformer.h)
    raise ValueError("Unknown model architecture")


def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    layer: int,
    model_type: str = "llama2",
    token_pos: int = -1,
) -> Tensor:
    """Extract activations at specified layer and token position."""
    model.eval()
    layers = get_layers(model)
    activations = []
    storage = [None]
    
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


def fit_ridge(H: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Fit ridge regression: w = (H'H + λI)^{-1} H'y, normalized."""
    D = H.shape[1]
    w = np.linalg.solve(H.T @ H + lam * np.eye(D), H.T @ y)
    return w / (np.linalg.norm(w) + 1e-8)


def fit_lat_probe(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    harmful_file: str,
    harmless_file: str,
    layer: int,
    output_dir: str,
    model_type: str = "llama2",
    lam: float = 1e-3,
    limit: Optional[int] = None,
) -> Tuple[Tensor, float]:
    """Fit LAT probe and save v_harm.pt. Returns (v_harm, auc)."""
    harmful = load_prompts(harmful_file, limit)
    harmless = load_prompts(harmless_file, limit)
    
    prompts = harmful + harmless
    labels = np.array([1.0] * len(harmful) + [0.0] * len(harmless))
    
    logger.info(f"Extracting {len(prompts)} prompts at layer {layer}")
    H = extract_activations(model, tokenizer, prompts, layer, model_type).numpy()
    
    w = fit_ridge(H, labels, lam)
    scores = H @ w
    auc = roc_auc_score(labels, scores)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    v_harm = torch.tensor(w, dtype=torch.float32)
    torch.save(v_harm, f"{output_dir}/v_harm.pt")
    
    # Save scores
    with open(f"{output_dir}/lat_scores.jsonl", 'w') as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({
                "prompt": p, "score": float(scores[i]),
                "label": "harmful" if labels[i] == 1 else "harmless"
            }) + '\n')
    
    logger.info(f"Saved v_harm.pt, AUC={auc:.4f}")
    return v_harm, auc


def score_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    v_harm: Tensor,
    layer: int,
    model_type: str = "llama2",
) -> np.ndarray:
    """Score prompts using trained LAT probe."""
    H = extract_activations(model, tokenizer, prompts, layer, model_type).numpy()
    return H @ v_harm.numpy()
