"""
Adaptive Safety Calibration - Main Pipeline

End-to-end orchestration:
1. Load model
2. Extract vectors (v_harm, v_refuse)
3. Choose optimal tau
4. Run inference with steering hooks
5. Evaluate results
"""
import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config
from utils import setup_logging, format_prompt, detect_refusal, load_prompts, save_jsonl
from extraction import fit_lat_probe, score_prompts
from extraction.refusal_vector import compute_refusal_vector
from steering import SteeringHooks
from evaluation import compute_metrics, find_optimal_tau, print_tau_analysis
from evaluation.metrics import print_metrics
from evaluation.choose_tau import load_scores

logger = logging.getLogger(__name__)


def load_model(config: Config):
    """Load model and tokenizer."""
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


def run_extraction(config: Config, model, tokenizer, harmful_file: str, harmless_file: str):
    """Extract v_harm vector via LAT probe."""
    logger.info("Step 1: Extracting harmfulness vector")
    v_harm, auc = fit_lat_probe(
        model, tokenizer,
        harmful_file, harmless_file,
        layer=config.l_lat,
        output_dir=config.output_dir,
        model_type=config.model_type,
        lam=config.ridge_lambda,
        limit=config.limit_prompts,
    )
    logger.info(f"LAT probe AUC: {auc:.4f}")
    return v_harm


def run_tau_selection(config: Config):
    """Find optimal tau from scores."""
    logger.info("Step 2: Selecting optimal tau")
    scores, labels = load_scores(config.scores_path)
    analysis = find_optimal_tau(scores, labels)
    print_tau_analysis(analysis)
    
    best = max(analysis["candidates"], key=lambda x: x["tradeoff_score"])
    logger.info(f"Selected tau={best['tau']:.4f} (tradeoff={best['tradeoff_score']:.2%})")
    return best["tau"]


def run_inference(
    config: Config,
    model,
    tokenizer,
    prompts: list,
    use_steering: bool = True,
) -> list:
    """Run inference, optionally with steering hooks."""
    results = []
    
    if use_steering:
        v_harm = torch.load(config.v_harm_path)
        v_refuse = torch.load(config.v_refuse_path)
        hooks = SteeringHooks(v_harm, v_refuse, tau=config.tau, alpha=config.alpha)
        hooks.register(model, l_lat=config.l_lat, l_post=config.l_post)
    
    try:
        for prompt in prompts:
            formatted = format_prompt(prompt, config.model_type)
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append({
                "prompt": prompt,
                "generation": response,
                "refused": detect_refusal(response),
            })
    finally:
        if use_steering:
            hooks.remove()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Adaptive Safety Calibration Pipeline")
    parser.add_argument("--extract", action="store_true", help="Run vector extraction")
    parser.add_argument("--harmful", type=str, help="Path to harmful prompts")
    parser.add_argument("--harmless", type=str, help="Path to harmless prompts")
    parser.add_argument("--infer", type=str, help="Path to prompts for inference")
    parser.add_argument("--tau", type=float, default=None, help="Override tau value")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    args = parser.parse_args()
    
    config = Config()
    if args.model:
        config.model_path = args.model
    if args.tau:
        config.tau = args.tau
    
    setup_logging(config.output_dir)
    model, tokenizer = load_model(config)
    
    if args.extract:
        if not args.harmful or not args.harmless:
            parser.error("--extract requires --harmful and --harmless")
        run_extraction(config, model, tokenizer, args.harmful, args.harmless)
        run_tau_selection(config)
    
    if args.infer:
        prompts = load_prompts(args.infer)
        logger.info(f"Running inference on {len(prompts)} prompts")
        
        results = run_inference(config, model, tokenizer, prompts, use_steering=True)
        save_jsonl(results, f"{config.output_dir}/steered_results.jsonl")
        
        metrics = compute_metrics(results)
        print_metrics(metrics, "Steered Results")


if __name__ == "__main__":
    main()
