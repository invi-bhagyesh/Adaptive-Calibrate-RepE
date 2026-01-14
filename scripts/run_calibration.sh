#!/bin/bash
# Run Calibration Script
# Runs the full calibration pipeline: inference, LAT fitting, and tau selection
#
# Usage: ./scripts/run_calibration.sh MODEL_PATH

set -e

MODEL_PTH=${1:-"NousResearch/Llama-2-7b-chat-hf"}
MODEL_TYPE=${2:-"llama2"}
DATA_DIR="data"
OUTPUT_DIR="outputs"

echo "================================================================"
echo "Running calibration pipeline"
echo "Model: $MODEL_PTH"
echo "Model Type: $MODEL_TYPE"
echo "================================================================"

# Step 1: Run inference on harmful dataset (AdvBench)
echo ""
echo "[Step 1/5] Running inference on harmful prompts..."
python scripts/run_inference.py \
    --model "$MODEL_PTH" \
    --input "${DATA_DIR}/advbench.json" \
    --output "${OUTPUT_DIR}/advbench_results.jsonl" \
    --model_type "$MODEL_TYPE" \
    --max_new_tokens 256

# Step 2: Run inference on harmless dataset (XSTest)
echo ""
echo "[Step 2/5] Running inference on harmless prompts..."
python scripts/run_inference.py \
    --model "$MODEL_PTH" \
    --input "${DATA_DIR}/xstest-harmless.json" \
    --output "${OUTPUT_DIR}/xstest_results.jsonl" \
    --model_type "$MODEL_TYPE" \
    --max_new_tokens 256

# Step 3: Check labeling
echo ""
echo "[Step 3/5] Checking refusal labeling..."
python scripts/tiny_check.py "${OUTPUT_DIR}/advbench_results.jsonl"
python scripts/tiny_check.py "${OUTPUT_DIR}/xstest_results.jsonl"

# Step 4: Fit LAT probe
echo ""
echo "[Step 4/5] Fitting LAT probe for v_harm..."
python extraction/lat_probe.py \
    --fit \
    --model "$MODEL_PTH" \
    --harmful_file "${DATA_DIR}/advbench.json" \
    --harmless_file "${DATA_DIR}/xstest-harmless.json" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --limit 200

# Step 5: Compute refusal vector
echo ""
echo "[Step 5a/5] Computing v_refuse (from XSTest results)..."
python extraction/refusal_vector.py \
    --model "$MODEL_PTH" \
    --results_file "${OUTPUT_DIR}/xstest_results.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE"

# Step 6: Choose tau
echo ""
echo "[Step 5b/5] Choosing optimal tau..."
python evaluation/choose_tau.py \
    --scores "${OUTPUT_DIR}/lat_scores.jsonl" \
    --output "${OUTPUT_DIR}/tau_analysis.json"

echo ""
echo "================================================================"
echo "Calibration complete!"
echo ""
echo "Generated files:"
echo "  - ${OUTPUT_DIR}/v_harm.pt"
echo "  - ${OUTPUT_DIR}/v_refuse.pt"
echo "  - ${OUTPUT_DIR}/lat_scores.jsonl"
echo "  - ${OUTPUT_DIR}/tau_analysis.json"
echo ""
echo "Next steps:"
echo "  1. Review tau_analysis.json for threshold candidates"
echo "  2. Run sweep_params.sh to find optimal operating point"
echo "  3. Run evaluation with chosen parameters"
echo "================================================================"
