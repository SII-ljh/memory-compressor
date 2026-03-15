#!/usr/bin/env bash
# Stage 1 — Baseline (no decoupled RoPE, no prompt bias)
set -euo pipefail

accelerate launch --num_processes 8 --multi_gpu \
  src/train.py --config config/default.yaml --stage 1 \
  --override \
    use_decoupled_rope=false \
    use_prompt_bias=false \
    output_dir=./outputs
