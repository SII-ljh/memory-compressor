#!/usr/bin/env bash
# Stage 1 Upper Bound: evaluate all Qwen3 models (direct NTP, no compression).
# Single H200 GPU.
#
# Usage:
#   bash scripts/benchmark/stage1_upper_bound_all.sh
set -euo pipefail

CONFIG="config/default.yaml"
OUTPUT_DIR="./outputs/benchmark"
BATCH_SIZE=4

MODELS=(
    "./models/Qwen3-0.6B"
    "./models/Qwen3-1.7B"
    "./models/Qwen3-4B"
)

echo "=============================================="
echo " Stage 1 Upper Bound: All Qwen3 Models"
echo "=============================================="

for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    OUTPUT_FILE="${OUTPUT_DIR}/stage1_upper_bound_${MODEL_NAME}.json"

    echo ""
    echo ">>> ${MODEL_NAME} ..."
    python scripts/benchmark/stage1_upper_bound.py \
        --config "$CONFIG" \
        --model_path "$MODEL_PATH" \
        --batch_size "$BATCH_SIZE" \
        --output "$OUTPUT_FILE"
    echo "    Saved: ${OUTPUT_FILE}"
done

echo ""
echo "=============================================="
echo " Done! Results:"
for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    echo "  ${OUTPUT_DIR}/stage1_upper_bound_${MODEL_NAME}.json"
done
echo "=============================================="
