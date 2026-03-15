#!/usr/bin/env bash
# Stage 2 — Decoupled RoPE only (loads stage 1 rope checkpoint)
set -euo pipefail

accelerate launch --num_processes 8 --multi_gpu \
  src/train.py --config config/default.yaml --stage 2 \
  --resume ./outputs/rope/stage1/best.pt \
  --override \
    use_decoupled_rope=true \
    use_prompt_bias=false \
    output_dir=./outputs
