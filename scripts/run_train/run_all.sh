#!/usr/bin/env bash
# Run all 5 Stage-1 ablation experiments sequentially.
#
# Experiment matrix (baseline perceiver, no RoPE, no bias):
#   Decoder ablation (fix M=128):  Qwen3-0.6B / 1.7B / 4B
#   Latent  ablation (fix 0.6B):   M=64 / 128 / 256
#   (0.6B + M=128 is shared → 5 unique runs)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  QCPC Stage-1 Ablation (8×H200)"
echo "=========================================="

# --- Latent length ablation (decoder = Qwen3-0.6B) ---
echo ""
echo "[1/5] Qwen3-0.6B  M=64"
bash "$SCRIPT_DIR/stage1_qwen06b_m64.sh"

echo ""
echo "[2/5] Qwen3-0.6B  M=128  (shared baseline)"
bash "$SCRIPT_DIR/stage1_qwen06b_m128.sh"

echo ""
echo "[3/5] Qwen3-0.6B  M=256"
bash "$SCRIPT_DIR/stage1_qwen06b_m256.sh"

# --- Decoder ablation (M = 128) ---
echo ""
echo "[4/5] Qwen3-1.7B  M=128"
bash "$SCRIPT_DIR/stage1_qwen17b_m128.sh"

echo ""
echo "[5/5] Qwen3-4B    M=128"
bash "$SCRIPT_DIR/stage1_qwen4b_m128.sh"

echo ""
echo "=========================================="
echo "  All 5 ablation experiments completed!"
echo "=========================================="
