#!/usr/bin/env bash
# Run Stage-1 and/or Stage-2 ablation experiments sequentially.
#
# Experiment matrix (baseline perceiver, no RoPE, no bias):
#   Decoder ablation (fix M=128):  Qwen3-0.6B / 1.7B / 4B
#   Latent  ablation (fix 0.6B):   M=64 / 128 / 256
#   (0.6B + M=128 is shared → 5 unique runs)
#
# Usage:
#   bash scripts/run_train/run_all.sh                                  # all experiments, stage 1+2
#   bash scripts/run_train/run_all.sh --stage 1                        # all experiments, stage 1 only
#   bash scripts/run_train/run_all.sh --exp 06b_m64,06b_m128           # only these experiments
#   bash scripts/run_train/run_all.sh --stage 1 --exp 06b_m64,4b_m128 # stage 1, selected experiments
#
# Available experiment names:
#   06b_m64  06b_m128  06b_m256  17b_m128  4b_m128
set -euo pipefail

STAGE="all"
EXPERIMENTS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage) STAGE="$2"; shift 2 ;;
        --exp)   EXPERIMENTS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# All experiments in default order
ALL_EXPS=("06b_m64" "06b_m128" "06b_m256" "17b_m128" "4b_m128")

# Parse --exp into array, or use all
if [[ -n "${EXPERIMENTS}" ]]; then
    IFS=',' read -ra EXPS <<< "${EXPERIMENTS}"
else
    EXPS=("${ALL_EXPS[@]}")
fi

# Map experiment name → display label
declare -A LABELS=(
    ["06b_m64"]="Qwen3-0.6B  M=64"
    ["06b_m128"]="Qwen3-0.6B  M=128  (shared baseline)"
    ["06b_m256"]="Qwen3-0.6B  M=256"
    ["17b_m128"]="Qwen3-1.7B  M=128"
    ["4b_m128"]="Qwen3-4B    M=128"
)

# Map experiment name → script suffix
declare -A SCRIPTS=(
    ["06b_m64"]="qwen06b_m64"
    ["06b_m128"]="qwen06b_m128"
    ["06b_m256"]="qwen06b_m256"
    ["17b_m128"]="qwen17b_m128"
    ["4b_m128"]="qwen4b_m128"
)

# Validate experiment names
for exp in "${EXPS[@]}"; do
    if [[ -z "${SCRIPTS[$exp]+x}" ]]; then
        echo "Error: unknown experiment '${exp}'"
        echo "Available: ${ALL_EXPS[*]}"
        exit 1
    fi
done

TOTAL=${#EXPS[@]}

# ─── Stage 1: Pretrain ──────────────────────────────────────────────
if [[ "${STAGE}" == "all" || "${STAGE}" == "1" ]]; then
    echo "=========================================="
    echo "  QCPC Stage-1 Pretrain (${TOTAL} experiments)"
    echo "=========================================="

    for i in "${!EXPS[@]}"; do
        exp="${EXPS[$i]}"
        echo ""
        echo "[S1 $((i+1))/${TOTAL}] ${LABELS[$exp]}"
        bash "$SCRIPT_DIR/stage1_${SCRIPTS[$exp]}.sh"
    done

    echo ""
    echo "  Stage 1 all done!"
    echo "=========================================="
fi

# ─── Stage 2: QA Finetune ───────────────────────────────────────────
if [[ "${STAGE}" == "all" || "${STAGE}" == "2" ]]; then
    echo "=========================================="
    echo "  QCPC Stage-2 QA Finetune (${TOTAL} experiments)"
    echo "=========================================="

    for i in "${!EXPS[@]}"; do
        exp="${EXPS[$i]}"
        echo ""
        echo "[S2 $((i+1))/${TOTAL}] ${LABELS[$exp]}"
        bash "$SCRIPT_DIR/stage2_${SCRIPTS[$exp]}.sh"
    done

    echo ""
    echo "  Stage 2 all done!"
    echo "=========================================="
fi
