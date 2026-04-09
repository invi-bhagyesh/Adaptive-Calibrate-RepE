#!/usr/bin/env python3
"""
Parameter Sweep Script

Sweeps over tau and alpha parameters to find the optimal operating point.

Usage:
    python scripts/sweep_params.py --model MODEL --v_harm v_harm.pt --v_refuse v_refuse.pt \
        --xstest data/xstest.json --advbench data/advbench.json
"""
import argparse
import json
import logging
from itertools import product
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from calibrate_repe.data import detect_refusal, format_prompt, load_prompts
from calibrate_repe.steering import SteeringHooks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_generation(model, tokenizer, prompts, model_type, max_new_tokens=128):
    """Generate responses for a list of prompt strings."""
    responses = []
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    for prompt in tqdm(prompts, desc="Generating", leave=False):
        formatted = format_prompt(prompt, model_type)
        inputs = tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_config)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if formatted in response:
            response = response[len(formatted):]
        responses.append(response.strip())
    return responses


def evaluate_params(model, tokenizer, xstest, advbench, v_harm, v_refuse,
                    tau, alpha, l_lat, l_post, model_type):
    """Evaluate a single (tau, alpha) configuration."""
    hooks = SteeringHooks(v_harm=v_harm, v_refuse=v_refuse, tau=tau, alpha=alpha)
    hooks.register(model, l_lat=l_lat, l_post=l_post)

    try:
        xstest_resp = run_generation(model, tokenizer, xstest, model_type)
        xstest_accepted = sum(1 for r in xstest_resp if not detect_refusal(r))
        compliance_rate = xstest_accepted / len(xstest) if xstest else 0

        advbench_resp = run_generation(model, tokenizer, advbench, model_type)
        advbench_refused = sum(1 for r in advbench_resp if detect_refusal(r))
        safety_rate = advbench_refused / len(advbench) if advbench else 0
    finally:
        hooks.remove()

    return {
        "tau": tau,
        "alpha": alpha,
        "compliance_rate": compliance_rate,
        "safety_rate": safety_rate,
        "tradeoff_score": (compliance_rate + safety_rate) / 2,
        "xstest_accepted": xstest_accepted,
        "xstest_total": len(xstest),
        "advbench_refused": advbench_refused,
        "advbench_total": len(advbench),
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep tau and alpha parameters")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="llama2",
                        choices=["llama2", "llama3", "qwen", "vicuna"])
    parser.add_argument("--v_harm", type=str, required=True)
    parser.add_argument("--v_refuse", type=str, required=True)
    parser.add_argument("--xstest", type=str, required=True)
    parser.add_argument("--advbench", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/sweep_results.json")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--tau_values", type=str, default="-0.5,-0.3,-0.1,0.0,0.1,0.3,0.5")
    parser.add_argument("--alpha_values", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--l_lat", type=int, default=None)
    parser.add_argument("--l_post", type=int, default=None)
    args = parser.parse_args()

    tau_values = [float(x) for x in args.tau_values.split(",")]
    alpha_values = [float(x) for x in args.alpha_values.split(",")]
    logger.info(f"Sweeping {len(tau_values)} tau x {len(alpha_values)} alpha values")

    logger.info(f"Loading model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    l_lat = args.l_lat if args.l_lat is not None else num_layers // 2
    l_post = args.l_post if args.l_post is not None else num_layers - 1
    logger.info(f"Using layers: L_lat={l_lat}, L_post={l_post}")

    v_harm = torch.load(args.v_harm, weights_only=True)
    v_refuse = torch.load(args.v_refuse, weights_only=True)

    xstest_prompts = load_prompts(args.xstest, args.limit)
    advbench_prompts = load_prompts(args.advbench, args.limit)
    logger.info(f"Loaded {len(xstest_prompts)} XSTest + {len(advbench_prompts)} AdvBench prompts")

    # Baseline (no intervention)
    logger.info("Running baseline (no intervention)...")
    baseline_xstest = run_generation(model, tokenizer, xstest_prompts, args.model_type)
    baseline_advbench = run_generation(model, tokenizer, advbench_prompts, args.model_type)
    baseline = {
        "tau": None, "alpha": None,
        "compliance_rate": sum(1 for r in baseline_xstest if not detect_refusal(r)) / len(xstest_prompts),
        "safety_rate": sum(1 for r in baseline_advbench if detect_refusal(r)) / len(advbench_prompts),
    }
    baseline["tradeoff_score"] = (baseline["compliance_rate"] + baseline["safety_rate"]) / 2

    # Sweep
    results = []
    best_result = None
    for tau, alpha in tqdm(list(product(tau_values, alpha_values)), desc="Sweeping"):
        result = evaluate_params(
            model, tokenizer, xstest_prompts, advbench_prompts,
            v_harm, v_refuse, tau, alpha, l_lat, l_post, args.model_type,
        )
        results.append(result)
        if best_result is None or result["tradeoff_score"] > best_result["tradeoff_score"]:
            best_result = result
        logger.info(f"tau={tau:.2f}, alpha={alpha:.1f} -> "
                     f"Comply: {result['compliance_rate']:.2%}, "
                     f"Safety: {result['safety_rate']:.2%}, "
                     f"Trade: {result['tradeoff_score']:.2%}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"baseline": baseline, "best_result": best_result, "all_results": results,
                    "config": {"model": args.model, "l_lat": l_lat, "l_post": l_post,
                               "tau_values": tau_values, "alpha_values": alpha_values}}, f, indent=2)

    print(f"\n{'='*60}")
    print("PARAMETER SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"\nBaseline:  Comply={baseline['compliance_rate']:.2%}  "
          f"Safety={baseline['safety_rate']:.2%}  Trade={baseline['tradeoff_score']:.2%}")
    print(f"Best:      tau={best_result['tau']:.4f}  alpha={best_result['alpha']:.2f}  "
          f"Comply={best_result['compliance_rate']:.2%}  "
          f"Safety={best_result['safety_rate']:.2%}  Trade={best_result['tradeoff_score']:.2%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
