#!/usr/bin/env bash
# Stage 2 Upper/Lower Bound: LoRA finetune + eval for all Qwen3 models.
# Single H200 GPU.  Each model: 1 epoch finetune → eval (upper + lower).
#
# Usage:
#   bash scripts/benchmark/stage2_finetune_eval_all.sh
#   bash scripts/benchmark/stage2_finetune_eval_all.sh --skip-finetune   # only eval
set -euo pipefail

CONFIG="config/default.yaml"
OUTPUT_DIR="./outputs/benchmark"
BATCH_SIZE=4
GRAD_ACCUM=8
EPOCHS=1
LR=2e-4
LORA_RANK=32
EVAL_INTERVAL=100
SKIP_FINETUNE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-finetune) SKIP_FINETUNE=true; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

MODELS=(
    "./models/Qwen3-0.6B"
    "./models/Qwen3-1.7B"
    "./models/Qwen3-4B"
)

echo "=============================================="
echo " Stage 2: LoRA Finetune + Eval (All Models)"
echo " Epochs: ${EPOCHS}  LR: ${LR}  LoRA rank: ${LORA_RANK}"
echo "=============================================="

for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    FT_OUTPUT="${OUTPUT_DIR}/stage2_finetuned_${MODEL_NAME}"
    LORA_PATH="${FT_OUTPUT}/best_lora"
    EVAL_OUTPUT="${OUTPUT_DIR}/stage2_eval_${MODEL_NAME}.json"

    echo ""
    echo "====== ${MODEL_NAME} ======"

    # --- Finetune ---
    if [[ "${SKIP_FINETUNE}" == "false" ]]; then
        echo ">>> [${MODEL_NAME}] LoRA finetune (${EPOCHS} epoch)..."
        python scripts/benchmark/stage2_finetune.py \
            --config "$CONFIG" \
            --model_path "$MODEL_PATH" \
            --epochs "$EPOCHS" \
            --lr "$LR" \
            --batch_size "$BATCH_SIZE" \
            --grad_accum "$GRAD_ACCUM" \
            --lora_rank "$LORA_RANK" \
            --eval_interval "$EVAL_INTERVAL" \
            --output_dir "$FT_OUTPUT"
    else
        echo ">>> [${MODEL_NAME}] Skipping finetune, using existing: ${LORA_PATH}"
    fi

    # --- Eval ---
    echo ">>> [${MODEL_NAME}] Eval (upper + lower bound)..."
    python scripts/benchmark/stage2_eval.py \
        --config "$CONFIG" \
        --model_path "$MODEL_PATH" \
        --lora_path "$LORA_PATH" \
        --mode both \
        --batch_size "$BATCH_SIZE" \
        --output "$EVAL_OUTPUT"
    echo "    Saved: ${EVAL_OUTPUT}"
done

echo ""
echo "=============================================="
echo " Done! Results:"
for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    echo "  ${OUTPUT_DIR}/stage2_eval_${MODEL_NAME}.json"
done
echo "=============================================="
