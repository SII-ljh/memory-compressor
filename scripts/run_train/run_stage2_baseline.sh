#!/usr/bin/env bash
# Stage 2 — Baseline (loads stage 1 baseline checkpoint)
set -euo pipefail

accelerate launch --num_processes 8 --multi_gpu \
  src/train.py --config config/default.yaml --stage 2 \
  --resume ./outputs/baseline/stage1/best.pt \
  --override \
    use_decoupled_rope=false \
    use_prompt_bias=false \
    output_dir=./outputs
