#!/usr/bin/env python3
"""
Run Inference Script

Generates model responses for input prompts and labels them with refusal status.

Usage:
    python scripts/run_inference.py --model MODEL_PATH --input INPUT_FILE --output OUTPUT_FILE
"""
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm

from calibrate_repe.data import (
    detect_refusal,
    extract_response,
    format_prompt,
    load_prompts_with_metadata,
    save_jsonl,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_inference(model, tokenizer, prompts, model_type, max_new_tokens=256):
    """Run inference on a list of prompt dicts. Returns results with generations."""
    results = []
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    model.eval()

    for item in tqdm(prompts, desc="Running inference"):
        formatted = format_prompt(item["prompt"], model_type)
        inputs = tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_config)

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = extract_response(full_output, model_type, formatted)

        result = {
            "prompt": item["prompt"],
            "generation": response,
            "refused": detect_refusal(response),
        }
        # Carry forward extra fields from input
        for key, value in item.items():
            if key not in result:
                result[key] = value
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference and label refusals")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="llama2",
                        choices=["llama2", "llama3", "qwen", "vicuna"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    logger.info(f"Loading model from {args.model}")
    device_map = "auto" if args.device == "auto" else args.device

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )

    prompts = load_prompts_with_metadata(args.input)
    results = run_inference(model, tokenizer, prompts, args.model_type, args.max_new_tokens)
    save_jsonl(results, args.output)

    refused_count = sum(1 for r in results if r["refused"])
    logger.info(f"Results saved to {args.output}")
    logger.info(f"Total: {len(results)}, Refused: {refused_count}, Accepted: {len(results) - refused_count}")


if __name__ == "__main__":
    main()
