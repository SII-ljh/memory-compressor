#!/bin/bash
# =============================================================================
# Benchmark: 测量 QCPC 两阶段压缩的上界和下界
#
# 前置: python scripts/data/data_processor.py --task all
#
# Stage 1 (NTP):
#   上界 = Qwen3 直接读全文做 NTP (无压缩)
#
# Stage 2 (QA):
#   上界 = LoRA 微调 Qwen3 + 完整 context 做 QA
#   下界 = LoRA 微调 Qwen3 只输入 question (无 context)
#
# Usage:
#   bash scripts/benchmark/run_all.sh
#   bash scripts/benchmark/run_all.sh --stage 1
#   bash scripts/benchmark/run_all.sh --stage 2 --skip-finetune
#   bash scripts/benchmark/run_all.sh --models 0.6B,1.7B
#   bash scripts/benchmark/run_all.sh --stage 2 --models 4B
# =============================================================================

set -euo pipefail

CONFIG="config/default.yaml"
OUTPUT_DIR="./outputs/benchmark"
SKIP_FINETUNE=false
STAGE="all"
MODELS_ARG="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-finetune) SKIP_FINETUNE=true; shift ;;
        --stage) STAGE="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --models) MODELS_ARG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# --- Resolve model list ---
declare -A MODEL_MAP=(
    ["0.6B"]="./models/Qwen3-0.6B"
    ["1.7B"]="./models/Qwen3-1.7B"
    ["4B"]="./models/Qwen3-4B"
)
ALL_KEYS=("0.6B" "1.7B" "4B")

if [[ "${MODELS_ARG}" == "all" ]]; then
    SELECTED_KEYS=("${ALL_KEYS[@]}")
else
    SELECTED_KEYS=()
    IFS=',' read -ra NAMES <<< "${MODELS_ARG}"
    for name in "${NAMES[@]}"; do
        name=$(echo "$name" | xargs)  # trim whitespace
        if [[ -z "${MODEL_MAP[$name]+x}" ]]; then
            echo "Error: Unknown model '$name'. Available: ${ALL_KEYS[*]}"
            exit 1
        fi
        SELECTED_KEYS+=("$name")
    done
fi

# --- Auto-detect GPU count ---
if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | xargs)
else
    NUM_GPUS=1
fi
if [[ ${NUM_GPUS} -lt 1 ]]; then NUM_GPUS=1; fi

echo "=============================================="
echo " QCPC Benchmark: Upper/Lower Bound"
echo "=============================================="
echo "Config: ${CONFIG}"
echo "Stage:  ${STAGE}"
echo "Models: ${SELECTED_KEYS[*]}"
echo "GPUs:   ${NUM_GPUS}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# ─── Stage 1 ─────────────────────────────────────────────────────────
if [[ "${STAGE}" == "all" || "${STAGE}" == "1" ]]; then
    for key in "${SELECTED_KEYS[@]}"; do
        MODEL_PATH="${MODEL_MAP[$key]}"
        MODEL_NAME="Qwen3-${key}"
        OUT_FILE="${OUTPUT_DIR}/stage1_upper_bound_${MODEL_NAME}.json"

        echo ">>> [${MODEL_NAME}] Stage 1 Upper Bound (Direct NTP)..."
        python scripts/benchmark/stage1_upper_bound.py \
            --config "${CONFIG}" \
            --model_path "${MODEL_PATH}" \
            --output "${OUT_FILE}"
        echo ""
    done
fi

# ─── Stage 2 ─────────────────────────────────────────────────────────
if [[ "${STAGE}" == "all" || "${STAGE}" == "2" ]]; then
    for key in "${SELECTED_KEYS[@]}"; do
        MODEL_PATH="${MODEL_MAP[$key]}"
        MODEL_NAME="Qwen3-${key}"
        FT_OUTPUT="${OUTPUT_DIR}/stage2_finetuned_${MODEL_NAME}"
        LORA_PATH="${FT_OUTPUT}/best_lora"
        EVAL_OUTPUT="${OUTPUT_DIR}/stage2_eval_${MODEL_NAME}.json"

        echo "====== ${MODEL_NAME} ======"

        # --- Finetune ---
        if [[ "${SKIP_FINETUNE}" == "false" ]]; then
            echo ">>> [${MODEL_NAME}] LoRA finetune (1 epoch)..."
            accelerate launch --num_processes "${NUM_GPUS}" --multi_gpu \
              scripts/benchmark/stage2_finetune.py \
                --config "${CONFIG}" \
                --model_path "${MODEL_PATH}" \
                --epochs 1 \
                --lr 2e-4 \
                --lora_rank 32 \
                --eval_interval 100 \
                --output_dir "${FT_OUTPUT}"
        else
            echo ">>> [${MODEL_NAME}] Skipping finetune, using existing: ${LORA_PATH}"
        fi

        # --- Eval ---
        echo ">>> [${MODEL_NAME}] Eval (upper + lower bound)..."
        python scripts/benchmark/stage2_eval.py \
            --config "${CONFIG}" \
            --model_path "${MODEL_PATH}" \
            --lora_path "${LORA_PATH}" \
            --mode both \
            --output "${EVAL_OUTPUT}"
        echo "    Saved: ${EVAL_OUTPUT}"
        echo ""
    done
fi

echo "=============================================="
echo " Done! Results:"
if [[ "${STAGE}" == "all" || "${STAGE}" == "1" ]]; then
    for key in "${SELECTED_KEYS[@]}"; do
        echo "  ${OUTPUT_DIR}/stage1_upper_bound_Qwen3-${key}.json"
    done
fi
if [[ "${STAGE}" == "all" || "${STAGE}" == "2" ]]; then
    for key in "${SELECTED_KEYS[@]}"; do
        echo "  ${OUTPUT_DIR}/stage2_eval_Qwen3-${key}.json"
    done
fi
echo "=============================================="
