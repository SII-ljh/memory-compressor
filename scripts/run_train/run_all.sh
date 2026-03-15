#!/usr/bin/env bash
# Run all 8 experiments sequentially: stage1 (4 modes) → stage2 (4 modes)
# Each stage2 automatically loads its corresponding stage1 best checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  QCPC Full Training Pipeline (8xH200)"
echo "=========================================="

# --- Stage 1: Pretrain all 4 modes ---
echo ""
echo "[1/8] Stage 1 — Baseline"
bash "$SCRIPT_DIR/run_stage1_baseline.sh"

echo ""
echo "[2/8] Stage 1 — Decoupled RoPE"
bash "$SCRIPT_DIR/run_stage1_rope.sh"

echo ""
echo "[3/8] Stage 1 — Prompt Bias"
bash "$SCRIPT_DIR/run_stage1_bias.sh"

echo ""
echo "[4/8] Stage 1 — Full Model"
bash "$SCRIPT_DIR/run_stage1_full.sh"

# --- Stage 2: QA finetune all 4 modes ---
echo ""
echo "[5/8] Stage 2 — Baseline"
bash "$SCRIPT_DIR/run_stage2_baseline.sh"

echo ""
echo "[6/8] Stage 2 — Decoupled RoPE"
bash "$SCRIPT_DIR/run_stage2_rope.sh"

echo ""
echo "[7/8] Stage 2 — Prompt Bias"
bash "$SCRIPT_DIR/run_stage2_bias.sh"

echo ""
echo "[8/8] Stage 2 — Full Model"
bash "$SCRIPT_DIR/run_stage2_full.sh"

echo ""
echo "=========================================="
echo "  All 8 experiments completed!"
echo "=========================================="
