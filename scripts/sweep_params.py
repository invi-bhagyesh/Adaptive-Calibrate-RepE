#!/usr/bin/env python3
"""
Parameter Sweep Script

Sweeps over tau and alpha parameters to find the optimal operating point
for the piece-wise steering intervention.

Usage:
    python scripts/sweep_params.py --model MODEL --v_harm v_harm.pt --v_refuse v_refuse.pt \
        --xstest data/xstest.json --advbench data/advbench.json --output sweep_results.json
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
from itertools import product

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "steering"))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from utils import formatInp_llama_persuasion, REFUSAL_PHRASE
from hooks import SteeringHooks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_refusal(response: str) -> bool:
    """Detect if a response is a refusal."""
    if not response:
        return False
    response_lower = response.lower()
    for phrase in REFUSAL_PHRASE:
        if phrase.lower() in response_lower:
            return True
    return False


def load_prompts(file_path: str, limit: int = None) -> List[str]:
    """Load prompts from a JSON file."""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            if isinstance(item, dict):
                prompt = item.get('instruction') or item.get('prompt') or item.get('question') or item.get('bad_q')
            else:
                prompt = item
            if prompt:
                prompts.append(prompt)
    
    if limit:
        prompts = prompts[:limit]
    return prompts


def run_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    model_type: str,
    max_new_tokens: int = 128,
) -> List[str]:
    """Run generation on a list of prompts."""
    responses = []
    
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    
    for prompt in tqdm(prompts, desc="Generating", leave=False):
        formatted = formatInp_llama_persuasion({'instruction': prompt}, model=model_type)
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_config)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the response part
        if formatted in response:
            response = response[len(formatted):]
        responses.append(response.strip())
    
    return responses


def evaluate_params(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    xstest_prompts: List[str],
    advbench_prompts: List[str],
    v_harm: torch.Tensor,
    v_refuse: torch.Tensor,
    tau: float,
    alpha: float,
    l_lat: int,
    l_post: int,
    model_type: str,
) -> Dict[str, float]:
    """
    Evaluate a single parameter configuration.
    
    Returns:
        Dictionary with compliance_rate, safety_rate, tradeoff_score
    """
    # Create and register hooks
    hooks = SteeringHooks(
        v_harm=v_harm,
        v_refuse=v_refuse,
        tau=tau,
        alpha=alpha,
    )
    hooks.register(model, l_lat=l_lat, l_post=l_post)
    
    try:
        # Run on XSTest (harmless - should be accepted)
        xstest_responses = run_generation(model, tokenizer, xstest_prompts, model_type)
        xstest_accepted = sum(1 for r in xstest_responses if not detect_refusal(r))
        compliance_rate = xstest_accepted / len(xstest_prompts) if xstest_prompts else 0
        
        # Run on AdvBench (harmful - should be refused)
        advbench_responses = run_generation(model, tokenizer, advbench_prompts, model_type)
        advbench_refused = sum(1 for r in advbench_responses if detect_refusal(r))
        safety_rate = advbench_refused / len(advbench_prompts) if advbench_prompts else 0
        
    finally:
        hooks.remove()
    
    tradeoff_score = (compliance_rate + safety_rate) / 2
    
    return {
        'tau': tau,
        'alpha': alpha,
        'compliance_rate': compliance_rate,
        'safety_rate': safety_rate,
        'tradeoff_score': tradeoff_score,
        'xstest_accepted': xstest_accepted,
        'xstest_total': len(xstest_prompts),
        'advbench_refused': advbench_refused,
        'advbench_total': len(advbench_prompts),
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep tau and alpha parameters")
    parser.add_argument('--model', type=str, required=True,
                       help="Model path")
    parser.add_argument('--model_type', type=str, default='llama2',
                       choices=['llama2', 'llama3', 'qwen', 'vicuna'])
    parser.add_argument('--v_harm', type=str, required=True,
                       help="Path to v_harm.pt")
    parser.add_argument('--v_refuse', type=str, required=True,
                       help="Path to v_refuse.pt")
    parser.add_argument('--xstest', type=str, required=True,
                       help="Path to XSTest (harmless) prompts")
    parser.add_argument('--advbench', type=str, required=True,
                       help="Path to AdvBench (harmful) prompts")
    parser.add_argument('--output', type=str, default='outputs/sweep_results.json',
                       help="Output file")
    parser.add_argument('--limit', type=int, default=50,
                       help="Limit prompts per dataset for speed")
    parser.add_argument('--tau_values', type=str, default='-0.5,-0.3,-0.1,0.0,0.1,0.3,0.5',
                       help="Comma-separated tau values to try")
    parser.add_argument('--alpha_values', type=str, default='0.5,1.0,1.5,2.0',
                       help="Comma-separated alpha values to try")
    parser.add_argument('--l_lat', type=int, default=None,
                       help="Layer for LAT (default: num_layers // 2)")
    parser.add_argument('--l_post', type=int, default=None,
                       help="Layer for post-inst (default: num_layers - 1)")
    
    args = parser.parse_args()
    
    # Parse parameter values
    tau_values = [float(x) for x in args.tau_values.split(',')]
    alpha_values = [float(x) for x in args.alpha_values.split(',')]
    
    logger.info(f"Sweeping {len(tau_values)} tau values x {len(alpha_values)} alpha values")
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    # Determine layers
    num_layers = model.config.num_hidden_layers
    l_lat = args.l_lat if args.l_lat is not None else num_layers // 2
    l_post = args.l_post if args.l_post is not None else num_layers - 1
    logger.info(f"Using layers: L_lat={l_lat}, L_post={l_post}")
    
    # Load vectors
    v_harm = torch.load(args.v_harm)
    v_refuse = torch.load(args.v_refuse)
    
    # Load prompts
    xstest_prompts = load_prompts(args.xstest, args.limit)
    advbench_prompts = load_prompts(args.advbench, args.limit)
    logger.info(f"Loaded {len(xstest_prompts)} XSTest + {len(advbench_prompts)} AdvBench prompts")
    
    # Run baseline first (no intervention)
    logger.info("Running baseline (no intervention)...")
    baseline_xstest = run_generation(model, tokenizer, xstest_prompts, args.model_type)
    baseline_advbench = run_generation(model, tokenizer, advbench_prompts, args.model_type)
    
    baseline = {
        'tau': None,
        'alpha': None,
        'compliance_rate': sum(1 for r in baseline_xstest if not detect_refusal(r)) / len(xstest_prompts),
        'safety_rate': sum(1 for r in baseline_advbench if detect_refusal(r)) / len(advbench_prompts),
    }
    baseline['tradeoff_score'] = (baseline['compliance_rate'] + baseline['safety_rate']) / 2
    
    logger.info(f"Baseline - Compliance: {baseline['compliance_rate']:.2%}, "
               f"Safety: {baseline['safety_rate']:.2%}, Tradeoff: {baseline['tradeoff_score']:.2%}")
    
    # Sweep parameters
    results = []
    best_result = None
    
    for tau, alpha in tqdm(list(product(tau_values, alpha_values)), desc="Sweeping"):
        result = evaluate_params(
            model, tokenizer, xstest_prompts, advbench_prompts,
            v_harm, v_refuse, tau, alpha, l_lat, l_post, args.model_type
        )
        results.append(result)
        
        if best_result is None or result['tradeoff_score'] > best_result['tradeoff_score']:
            best_result = result
        
        logger.info(f"tau={tau:.2f}, alpha={alpha:.1f} -> "
                   f"Comply: {result['compliance_rate']:.2%}, "
                   f"Safety: {result['safety_rate']:.2%}, "
                   f"Trade: {result['tradeoff_score']:.2%}")
    
    # Save results
    output_data = {
        'baseline': baseline,
        'best_result': best_result,
        'all_results': results,
        'config': {
            'model': args.model,
            'l_lat': l_lat,
            'l_post': l_post,
            'tau_values': tau_values,
            'alpha_values': alpha_values,
        }
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("PARAMETER SWEEP RESULTS")
    print("="*60)
    print(f"\nBaseline:")
    print(f"  Compliance: {baseline['compliance_rate']:.2%}")
    print(f"  Safety:     {baseline['safety_rate']:.2%}")
    print(f"  Tradeoff:   {baseline['tradeoff_score']:.2%}")
    print(f"\nBest Configuration:")
    print(f"  tau={best_result['tau']:.4f}, alpha={best_result['alpha']:.2f}")
    print(f"  Compliance: {best_result['compliance_rate']:.2%}")
    print(f"  Safety:     {best_result['safety_rate']:.2%}")
    print(f"  Tradeoff:   {best_result['tradeoff_score']:.2%}")
    print("="*60 + "\n")
    
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
