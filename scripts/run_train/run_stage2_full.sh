#!/usr/bin/env bash
# Stage 2 — Full Model (loads stage 1 full checkpoint)
set -euo pipefail

accelerate launch --num_processes 8 --multi_gpu \
  src/train.py --config config/default.yaml --stage 2 \
  --resume ./outputs/full/stage1/best.pt \
  --override \
    use_decoupled_rope=true \
    use_prompt_bias=true \
    output_dir=./outputs
