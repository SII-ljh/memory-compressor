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
#   bash scripts/benchmark/run_all.sh --skip-finetune
#   bash scripts/benchmark/run_all.sh --stage 1
#   bash scripts/benchmark/run_all.sh --stage 2
# =============================================================================

set -euo pipefail

CONFIG="config/default.yaml"
OUTPUT_DIR="./outputs/benchmark"
SKIP_FINETUNE=false
STAGE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-finetune) SKIP_FINETUNE=true; shift ;;
        --stage) STAGE="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo " QCPC Benchmark: Upper/Lower Bound"
echo "=============================================="
echo "Config: ${CONFIG}"
echo "Stage:  ${STAGE}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# ─── Stage 1 ─────────────────────────────────────────────────────────
if [[ "${STAGE}" == "all" || "${STAGE}" == "1" ]]; then
    echo ">>> Stage 1 Upper Bound (Qwen3 Direct NTP)..."
    python scripts/benchmark/stage1_upper_bound.py \
        --config "${CONFIG}" \
        --batch_size 4 \
        --output "${OUTPUT_DIR}/stage1_upper_bound.json"
    echo ""
fi

# ─── Stage 2 ─────────────────────────────────────────────────────────
if [[ "${STAGE}" == "all" || "${STAGE}" == "2" ]]; then
    if [[ "${SKIP_FINETUNE}" == "false" ]]; then
        echo ">>> Stage 2 Fine-tune (LoRA on Qwen3)..."
        python scripts/benchmark/stage2_finetune.py \
            --config "${CONFIG}" \
            --epochs 3 \
            --lr 2e-4 \
            --batch_size 4 \
            --grad_accum 8 \
            --lora_rank 32 \
            --eval_interval 100 \
            --output_dir "${OUTPUT_DIR}/stage2_finetuned"
        echo ""
    fi

    echo ">>> Stage 2 Eval (Upper + Lower)..."
    python scripts/benchmark/stage2_eval.py \
        --config "${CONFIG}" \
        --lora_path "${OUTPUT_DIR}/stage2_finetuned/best_lora" \
        --mode both \
        --batch_size 4 \
        --output "${OUTPUT_DIR}/stage2_eval_results.json"
    echo ""
fi

echo "=============================================="
echo " Done! Results:"
[[ "${STAGE}" == "all" || "${STAGE}" == "1" ]] && echo "  ${OUTPUT_DIR}/stage1_upper_bound.json"
[[ "${STAGE}" == "all" || "${STAGE}" == "2" ]] && echo "  ${OUTPUT_DIR}/stage2_eval_results.json"
echo "=============================================="
