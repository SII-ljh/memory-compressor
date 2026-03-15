#!/bin/bash
# =============================================================================
# Run all benchmark scripts: establish upper/lower bounds for QCPC compression.
#
# Stage 1 (Pretraining - NTP):
#   Upper bound = Qwen3 直接读全文做 NTP (无压缩，最优性能)
#
# Stage 2 (SFT - QA):
#   Upper bound = LoRA 微调后的 Qwen3 + 完整 context 做 QA
#   Lower bound = LoRA 微调后的 Qwen3 只输入 question (无 context)
#
# QCPC 的性能应落在 upper 和 lower bound 之间。
#
# Usage:
#   bash scripts/benchmark/run_all.sh
#   bash scripts/benchmark/run_all.sh --skip-finetune   # skip stage 2 training
#   bash scripts/benchmark/run_all.sh --stage 1         # only stage 1
#   bash scripts/benchmark/run_all.sh --stage 2         # only stage 2
# =============================================================================

set -euo pipefail

CONFIG="config/default.yaml"
OUTPUT_DIR="./outputs/benchmark"
SKIP_FINETUNE=false
STAGE="all"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-finetune) SKIP_FINETUNE=true; shift ;;
        --stage) STAGE="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo " QCPC Benchmark: Upper/Lower Bound Evaluation"
echo "=============================================="
echo "Config:  ${CONFIG}"
echo "Output:  ${OUTPUT_DIR}"
echo "Stage:   ${STAGE}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# ─── Step 0: Prepare evaluation data ─────────────────────────────────
echo ">>> Step 0: Preparing evaluation data splits..."
python scripts/benchmark/prepare_eval_data.py \
    --data_root ./data \
    --output_root ./data/processed/sft \
    --eval_ratio 0.10 \
    --dev_ratio 0.05
echo ""

# ─── Step 1: Stage 1 Upper Bound ─────────────────────────────────────
if [[ "${STAGE}" == "all" || "${STAGE}" == "1" ]]; then
    echo ">>> Step 1: Stage 1 Upper Bound (Qwen3 Direct NTP)..."
    python scripts/benchmark/stage1_upper_bound.py \
        --config "${CONFIG}" \
        --eval_data ./data/processed/pretrain_eval.jsonl \
        --max_samples 500 \
        --batch_size 4 \
        --output "${OUTPUT_DIR}/stage1_upper_bound.json"
    echo ""
fi

# ─── Step 2: Stage 2 Fine-tune ──────────────────────────────────────
if [[ "${STAGE}" == "all" || "${STAGE}" == "2" ]]; then
    if [[ "${SKIP_FINETUNE}" == "false" ]]; then
        echo ">>> Step 2a: Stage 2 Fine-tune (LoRA on Qwen3)..."
        python scripts/benchmark/stage2_finetune.py \
            --config "${CONFIG}" \
            --train_data ./data/processed/sft/train.json \
            --dev_data ./data/processed/sft/dev.json \
            --epochs 3 \
            --lr 2e-4 \
            --batch_size 4 \
            --grad_accum 8 \
            --lora_rank 32 \
            --eval_interval 100 \
            --output_dir "${OUTPUT_DIR}/stage2_finetuned"
        echo ""
    else
        echo ">>> Skipping Stage 2 fine-tuning (--skip-finetune)"
        echo ""
    fi

    # ─── Step 3: Stage 2 Eval (Upper + Lower) ───────────────────────
    echo ">>> Step 2b: Stage 2 Evaluation (Upper & Lower Bounds)..."
    python scripts/benchmark/stage2_eval.py \
        --config "${CONFIG}" \
        --lora_path "${OUTPUT_DIR}/stage2_finetuned/best_lora" \
        --eval_data ./data/processed/sft/eval.json \
        --mode both \
        --batch_size 4 \
        --max_new_tokens 256 \
        --output "${OUTPUT_DIR}/stage2_eval_results.json"
    echo ""
fi

# ─── Summary ─────────────────────────────────────────────────────────
echo "=============================================="
echo " Benchmark Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
[[ "${STAGE}" == "all" || "${STAGE}" == "1" ]] && echo "  Stage 1 Upper Bound: ${OUTPUT_DIR}/stage1_upper_bound.json"
[[ "${STAGE}" == "all" || "${STAGE}" == "2" ]] && echo "  Stage 2 Fine-tune:   ${OUTPUT_DIR}/stage2_finetuned/training_info.json"
[[ "${STAGE}" == "all" || "${STAGE}" == "2" ]] && echo "  Stage 2 Eval:        ${OUTPUT_DIR}/stage2_eval_results.json"
echo ""
echo "Eval-only dataset:     ./data/processed/sft/eval.json"
echo ""
echo "After training QCPC, compare its metrics against these bounds."
