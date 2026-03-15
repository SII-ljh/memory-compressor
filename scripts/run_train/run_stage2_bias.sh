#!/usr/bin/env bash
# Stage 2 — Prompt Bias only (loads stage 1 bias checkpoint)
set -euo pipefail

accelerate launch --num_processes 8 --multi_gpu \
  src/train.py --config config/default.yaml --stage 2 \
  --resume ./outputs/bias/stage1/best.pt \
  --override \
    use_decoupled_rope=false \
    use_prompt_bias=true \
    output_dir=./outputs
