#!/usr/bin/env python3
"""
Tiny Check — quick sanity check for refusal labeling in inference outputs.

Usage:
    python scripts/tiny_check.py RESULTS_FILE
"""
import json
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/tiny_check.py <results_file.jsonl>")
        sys.exit(1)

    fn = sys.argv[1]
    items = []
    with open(fn, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    refused = sum(1 for item in items if item.get("refused", False))
    accepted = len(items) - refused

    print(f"\n{'='*60}")
    print(f"File: {fn}")
    print(f"Total: {len(items)}  Refused: {refused} ({100*refused/len(items):.1f}%)  "
          f"Accepted: {accepted} ({100*accepted/len(items):.1f}%)")
    print(f"{'='*60}\n")

    print("Sample entries:")
    print("-" * 60)
    for i, item in enumerate(items[:5]):
        prompt = item.get("prompt", "")[:100]
        if len(item.get("prompt", "")) > 100:
            prompt += "..."
        status = "REFUSED" if item.get("refused", False) else "ACCEPTED"
        print(f"[{i+1}] {status}: {prompt}")
        gen = item.get("generation", "")[:80]
        if gen:
            if len(item.get("generation", "")) > 80:
                gen += "..."
            print(f"    Response: {gen}")
        print()
    print("-" * 60)


if __name__ == "__main__":
    main()
