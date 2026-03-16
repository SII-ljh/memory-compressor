#!/usr/bin/env bash
# Run Stage-1 and/or Stage-2 ablation experiments sequentially.
#
# Experiment matrix (baseline perceiver, no RoPE, no bias):
#   Decoder ablation (fix M=128):  Qwen3-0.6B / 1.7B / 4B
#   Latent  ablation (fix 0.6B):   M=64 / 128 / 256
#   (0.6B + M=128 is shared → 5 unique runs)
#
# Usage:
#   bash scripts/run_train/run_all.sh              # stage 1 + 2
#   bash scripts/run_train/run_all.sh --stage 1    # stage 1 only
#   bash scripts/run_train/run_all.sh --stage 2    # stage 2 only
set -euo pipefail

STAGE="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage) STAGE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Stage 1: Pretrain ──────────────────────────────────────────────
if [[ "${STAGE}" == "all" || "${STAGE}" == "1" ]]; then
    echo "=========================================="
    echo "  QCPC Stage-1 Pretrain (8×H200)"
    echo "=========================================="

    echo ""
    echo "[S1 1/5] Qwen3-0.6B  M=64"
    bash "$SCRIPT_DIR/stage1_qwen06b_m64.sh"

    echo ""
    echo "[S1 2/5] Qwen3-0.6B  M=128  (shared baseline)"
    bash "$SCRIPT_DIR/stage1_qwen06b_m128.sh"

    echo ""
    echo "[S1 3/5] Qwen3-0.6B  M=256"
    bash "$SCRIPT_DIR/stage1_qwen06b_m256.sh"

    echo ""
    echo "[S1 4/5] Qwen3-1.7B  M=128"
    bash "$SCRIPT_DIR/stage1_qwen17b_m128.sh"

    echo ""
    echo "[S1 5/5] Qwen3-4B    M=128"
    bash "$SCRIPT_DIR/stage1_qwen4b_m128.sh"

    echo ""
    echo "  Stage 1 all done!"
    echo "=========================================="
fi

# ─── Stage 2: QA Finetune ───────────────────────────────────────────
if [[ "${STAGE}" == "all" || "${STAGE}" == "2" ]]; then
    echo "=========================================="
    echo "  QCPC Stage-2 QA Finetune (8×H200)"
    echo "=========================================="

    echo ""
    echo "[S2 1/5] Qwen3-0.6B  M=64"
    bash "$SCRIPT_DIR/stage2_qwen06b_m64.sh"

    echo ""
    echo "[S2 2/5] Qwen3-0.6B  M=128  (shared baseline)"
    bash "$SCRIPT_DIR/stage2_qwen06b_m128.sh"

    echo ""
    echo "[S2 3/5] Qwen3-0.6B  M=256"
    bash "$SCRIPT_DIR/stage2_qwen06b_m256.sh"

    echo ""
    echo "[S2 4/5] Qwen3-1.7B  M=128"
    bash "$SCRIPT_DIR/stage2_qwen17b_m128.sh"

    echo ""
    echo "[S2 5/5] Qwen3-4B    M=128"
    bash "$SCRIPT_DIR/stage2_qwen4b_m128.sh"

    echo ""
    echo "  Stage 2 all done!"
    echo "=========================================="
fi
