#!/bin/bash
# Sweep Parameters Script
# Sweeps over tau and alpha to find optimal operating point
#
# Usage: ./scripts/sweep_params.sh MODEL_PATH

MODEL_PTH=${1:-"NousResearch/Llama-2-7b-chat-hf"}
MODEL_TYPE=${2:-"llama2"}
OUTPUT_DIR="outputs"

echo "================================================================"
echo "Running parameter sweep"
echo "Model: $MODEL_PTH"
echo "================================================================"

python scripts/sweep_params.py \
    --model "$MODEL_PTH" \
    --model_type "$MODEL_TYPE" \
    --v_harm "${OUTPUT_DIR}/v_harm.pt" \
    --v_refuse "${OUTPUT_DIR}/v_refuse.pt" \
    --xstest "data/xstest-harmless.json" \
    --advbench "data/advbench.json" \
    --output "${OUTPUT_DIR}/sweep_results.json" \
    --limit 50 \
    --tau_values "-0.5,-0.3,-0.1,0.0,0.1,0.3,0.5" \
    --alpha_values "0.5,1.0,1.5,2.0"

echo ""
echo "Sweep complete! Results saved to ${OUTPUT_DIR}/sweep_results.json"
