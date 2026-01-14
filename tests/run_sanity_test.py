#!/usr/bin/env python3
"""
Sanity Test Script

Runs a basic test of the steering intervention on a small set of prompts.
Compares baseline vs steered outputs.

Usage:
    python tests/run_sanity_test.py --model MODEL --v_harm v_harm.pt --v_refuse v_refuse.pt --tau 0.5 --alpha 1.0
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "steering"))

from utils import formatInp_llama_persuasion, REFUSAL_PHRASE
from hooks import SteeringHooks

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
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


def generate_response(model, tokenizer, prompt, model_type, max_tokens=128):
    """Generate a single response."""
    formatted = formatInp_llama_persuasion({'instruction': prompt}, model=model_type)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    gen_config = GenerationConfig(
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if formatted in response:
        response = response[len(formatted):]
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Run sanity test")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='llama2')
    parser.add_argument('--v_harm', type=str, required=True)
    parser.add_argument('--v_refuse', type=str, required=True)
    parser.add_argument('--tau', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--prompts', type=str, default='tests/sanity_prompts.json')
    parser.add_argument('--output', type=str, default='outputs/sanity_test_results.json')
    
    args = parser.parse_args()
    
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
    
    # Load vectors
    v_harm = torch.load(args.v_harm)
    v_refuse = torch.load(args.v_refuse)
    
    # Determine layers
    num_layers = model.config.num_hidden_layers
    l_lat = num_layers // 2
    l_post = num_layers - 1
    
    # Load test prompts
    with open(args.prompts, 'r') as f:
        prompts = json.load(f)
    
    results = []
    
    print("\n" + "="*80)
    print("SANITY TEST RESULTS")
    print("="*80)
    
    for item in prompts:
        prompt = item['prompt']
        category = item.get('category', 'unknown')
        expected_refusal = item.get('expected_refusal', None)
        
        # Baseline
        baseline_response = generate_response(model, tokenizer, prompt, args.model_type)
        baseline_refused = detect_refusal(baseline_response)
        
        # Steered
        hooks = SteeringHooks(v_harm, v_refuse, tau=args.tau, alpha=args.alpha)
        hooks.register(model, l_lat=l_lat, l_post=l_post)
        
        steered_response = generate_response(model, tokenizer, prompt, args.model_type)
        steered_refused = detect_refusal(steered_response)
        
        hooks.remove()
        
        # Record result
        result = {
            'prompt': prompt,
            'category': category,
            'expected_refusal': expected_refusal,
            'baseline_response': baseline_response[:200],
            'baseline_refused': baseline_refused,
            'steered_response': steered_response[:200],
            'steered_refused': steered_refused,
            'behavior_changed': baseline_refused != steered_refused,
        }
        results.append(result)
        
        # Print result
        status = "✓" if (expected_refusal is None or 
                        (category == 'harmful' and steered_refused) or
                        (category == 'harmless' and not steered_refused)) else "✗"
        
        print(f"\n{status} [{category.upper()}] {prompt[:60]}...")
        print(f"  Baseline: {'REFUSED' if baseline_refused else 'ACCEPTED'}")
        print(f"  Steered:  {'REFUSED' if steered_refused else 'ACCEPTED'}")
        if result['behavior_changed']:
            print(f"  → Behavior changed!")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    harmful_correct = sum(1 for r in results if r['category'] == 'harmful' and r['steered_refused'])
    harmful_total = sum(1 for r in results if r['category'] == 'harmful')
    harmless_correct = sum(1 for r in results if r['category'] == 'harmless' and not r['steered_refused'])
    harmless_total = sum(1 for r in results if r['category'] == 'harmless')
    
    print(f"\nHarmful prompts refused:   {harmful_correct}/{harmful_total}")
    print(f"Harmless prompts accepted: {harmless_correct}/{harmless_total}")
    print(f"Behavior changes:          {sum(1 for r in results if r['behavior_changed'])}/{len(results)}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({'config': vars(args), 'results': results}, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
