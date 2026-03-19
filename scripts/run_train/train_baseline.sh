#!/usr/bin/env bash
# QCPC Baseline Training: learnable PE + pure learnable latent (no prompt bias)
#
# Usage:
#   bash scripts/run_train/train_baseline.sh                       # Full pipeline: 1a → 1b → 2
#   bash scripts/run_train/train_baseline.sh --stage 1             # Stage 1 only (1a + 1b)
#   bash scripts/run_train/train_baseline.sh --stage 1a            # Stage 1a only
#   bash scripts/run_train/train_baseline.sh --stage 1b            # Stage 1b only
#   bash scripts/run_train/train_baseline.sh --stage 2             # Stage 2 only
#
# Environment variables:
#   NUM_GPUS     — number of GPUs (auto-detected if unset)
#   QWEN_MODEL   — path to Qwen3 model (default: ./models/Qwen3-0.6B)
#   M            — number of memory tokens (default: 128)
#   OUTPUT_DIR   — output directory (default: ./outputs/baseline)
set -euo pipefail

STAGE="${1:---stage}"
[[ "${STAGE}" == "--stage" ]] && { STAGE="${2:-all}"; } || true
# Parse --stage arg
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage) STAGE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | xargs)}
[[ ${NUM_GPUS} -lt 1 ]] && NUM_GPUS=1

QWEN_MODEL=${QWEN_MODEL:-./models/Qwen3-0.6B}
M=${M:-128}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/baseline}

echo "=========================================="
echo "  QCPC Baseline (no prompt bias)"
echo "  Model: ${QWEN_MODEL}"
echo "  M=${M}, GPUs=${NUM_GPUS}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Stage: ${STAGE}"
echo "=========================================="

run_stage() {
    local stage=$1
    echo ""
    echo "[Stage ${stage}] Starting..."
    accelerate launch --num_processes "${NUM_GPUS}" --multi_gpu \
      src/train.py --config config/default.yaml --stage "${stage}" \
      --override \
        use_prompt_bias=false \
        qwen3_model_path="${QWEN_MODEL}" \
        num_memory_tokens="${M}" \
        output_dir="${OUTPUT_DIR}"
    echo "[Stage ${stage}] Done."
}

case "${STAGE}" in
    all)
        run_stage 1a
        run_stage 1b
        run_stage 2
        ;;
    1)
        run_stage 1a
        run_stage 1b
        ;;
    1a|1b|2)
        run_stage "${STAGE}"
        ;;
    *)
        echo "Unknown stage: ${STAGE}. Use: all, 1, 1a, 1b, 2"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "  Baseline training complete!"
echo "=========================================="
