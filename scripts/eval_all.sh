#!/usr/bin/env bash
# Evaluate all experiments in outputs/ (excluding benchmark).
# For each experiment, evaluate stage1b and stage2 best checkpoints.
# Results are collected into eval_results.txt with a summary table.
#
# Usage:
#   bash scripts/eval_all.sh                           # all experiments, stage1b + stage2
#   bash scripts/eval_all.sh --stages 2                # stage2 only
#   bash scripts/eval_all.sh --exp qwen06b_m64,qwen06b_m128
#   bash scripts/eval_all.sh --gen_samples 500         # more generation samples
#   bash scripts/eval_all.sh --num_gpus 4              # fewer GPUs
#   bash scripts/eval_all.sh --output my_results.txt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ── Defaults ─────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-8}
GEN_SAMPLES=200
BATCH_SIZE=8
EXPERIMENTS=""
STAGES="1b,2"
OUTPUT_FILE="eval_results.txt"

# ── Parse args ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)     NUM_GPUS="$2";     shift 2 ;;
        --gen_samples)  GEN_SAMPLES="$2";  shift 2 ;;
        --batch_size)   BATCH_SIZE="$2";   shift 2 ;;
        --exp)          EXPERIMENTS="$2";  shift 2 ;;
        --stages)       STAGES="$2";       shift 2 ;;
        --output)       OUTPUT_FILE="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Parse stages
IFS=',' read -ra STAGE_LIST <<< "${STAGES}"

# ── Discover experiments ─────────────────────────────────────
if [[ -n "${EXPERIMENTS}" ]]; then
    IFS=',' read -ra EXP_DIRS <<< "${EXPERIMENTS}"
else
    EXP_DIRS=()
    for d in outputs/*/; do
        [[ ! -d "$d" ]] && continue
        name=$(basename "$d")
        # Exclude benchmark directory
        [[ "$name" == "benchmark" ]] && continue
        EXP_DIRS+=("$name")
    done
fi

if [[ ${#EXP_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: No experiment directories found in outputs/"
    echo "Expected structure: outputs/<experiment>/stage1b/best.pt"
    exit 1
fi

# Sort experiment names for consistent ordering
IFS=$'\n' EXP_DIRS=($(sort <<<"${EXP_DIRS[*]}")); unset IFS

echo "=========================================="
echo "  QCPC Batch Evaluation"
echo "  Experiments: ${#EXP_DIRS[@]}"
echo "  Stages: ${STAGES}"
echo "  GPUs: ${NUM_GPUS}"
echo "  Gen samples: ${GEN_SAMPLES}"
echo "  Output: ${OUTPUT_FILE}"
echo "=========================================="
echo ""

# ── Prepare output dirs ─────────────────────────────────────
RESULTS_DIR="outputs/eval_results"
LOGS_DIR="${RESULTS_DIR}/logs"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Write header
{
    echo "QCPC Evaluation Results"
    echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "GPUs: ${NUM_GPUS}, Gen samples: ${GEN_SAMPLES}, Batch size: ${BATCH_SIZE}"
    echo ""
} > "$OUTPUT_FILE"

# ── Evaluate each experiment ─────────────────────────────────
TOTAL=${#EXP_DIRS[@]}

for idx in "${!EXP_DIRS[@]}"; do
    exp="${EXP_DIRS[$idx]}"
    echo ""
    echo "[$((idx+1))/${TOTAL}] ──── ${exp} ────"

    EXP_DIR="outputs/${exp}"

    for stage in "${STAGE_LIST[@]}"; do
        CKPT="${EXP_DIR}/${stage}/best.pt"

        if [[ ! -f "$CKPT" ]]; then
            echo "  [${stage}] SKIP (no checkpoint: ${CKPT})"
            echo "[${exp}] Stage ${stage}: SKIPPED (no checkpoint)" >> "$OUTPUT_FILE"
            continue
        fi

        echo "  [${stage}] Evaluating ${CKPT} ..."
        JSON_OUT="${RESULTS_DIR}/${exp}_${stage}.json"
        LOG_OUT="${LOGS_DIR}/${exp}_${stage}.log"

        # Build accelerate command
        CMD=(
            accelerate launch
            --num_processes "${NUM_GPUS}"
            --multi_gpu
            src/evaluate.py
            --checkpoint "$CKPT"
            --stage "${stage}"
            --auto_config
            --batch_size "${BATCH_SIZE}"
            --output_json "$JSON_OUT"
        )

        # Add generation samples for stage 2
        if [[ "$stage" == "2" ]]; then
            CMD+=(--gen_samples "${GEN_SAMPLES}")
        fi

        # Run evaluation, capture log
        if "${CMD[@]}" > "$LOG_OUT" 2>&1; then
            echo "  [${stage}] Done. Results: ${JSON_OUT}"

            # Append to results file
            echo "[${exp}] Stage ${stage}:" >> "$OUTPUT_FILE"
            cat "$JSON_OUT" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
        else
            echo "  [${stage}] FAILED (see ${LOG_OUT})"
            echo "[${exp}] Stage ${stage}: FAILED (see ${LOG_OUT})" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
        fi
    done
done

# ── Generate summary table ───────────────────────────────────
{
    echo ""
    echo "================================ SUMMARY ================================"
    printf "%-20s %-8s %10s %10s %10s %10s %10s\n" \
        "Experiment" "Stage" "Loss" "PPL" "ROUGE-L" "F1" "EM"
    echo "-------------------------------------------------------------------------"

    for exp in "${EXP_DIRS[@]}"; do
        for stage in "${STAGE_LIST[@]}"; do
            JSON_FILE="${RESULTS_DIR}/${exp}_${stage}.json"
            if [[ -f "$JSON_FILE" ]]; then
                # Extract metrics from JSON
                read -r LOSS PPL ROUGE F1 EM < <(
                    python3 -c "
import json, sys
d = json.load(open('${JSON_FILE}'))
print(d.get('loss', '-'), d.get('ppl', '-'), d.get('rouge_l', '-'), d.get('f1', '-'), d.get('em', '-'))
"
                )
                printf "%-20s %-8s %10s %10s %10s %10s %10s\n" \
                    "$exp" "$stage" "$LOSS" "$PPL" "$ROUGE" "$F1" "$EM"
            else
                printf "%-20s %-8s %10s %10s %10s %10s %10s\n" \
                    "$exp" "$stage" "SKIP" "SKIP" "-" "-" "-"
            fi
        done
    done

    echo "========================================================================="
} >> "$OUTPUT_FILE"

# ── Print final summary ──────────────────────────────────────
echo ""
echo "=========================================="
echo "  Evaluation complete!"
echo "  Results:  ${OUTPUT_FILE}"
echo "  JSON:     ${RESULTS_DIR}/"
echo "  Logs:     ${LOGS_DIR}/"
echo "=========================================="
echo ""
cat "$OUTPUT_FILE"
