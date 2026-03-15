#!/usr/bin/env bash
# Stage 1 — Full Model (decoupled RoPE + prompt bias)
set -euo pipefail

accelerate launch --num_processes 8 --multi_gpu \
  src/train.py --config config/default.yaml --stage 1 \
  --override \
    use_decoupled_rope=true \
    use_prompt_bias=true \
    output_dir=./outputs
