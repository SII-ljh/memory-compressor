#!/usr/bin/env bash
# ============================================================================
# Run ALL Stage-1a (short window warmup) experiments in PARALLEL on 8x H200.
#
# GPU allocation strategy (8 GPUs total):
#   GPU 0     : Qwen3-0.6B  M=64   (single GPU, model is tiny)
#   GPU 1     : Qwen3-0.6B  M=128  (single GPU)
#   GPU 2     : Qwen3-0.6B  M=256  (single GPU)
#   GPU 3-4   : Qwen3-1.7B  M=128  (2 GPUs, DDP for throughput)
#   GPU 5-7   : Qwen3-4B    M=128  (3 GPUs, DDP for throughput)
#
# Stage 1a is short-context (512 tokens), so 0.6B models easily fit on 1 GPU
# with large auto-detected batch sizes. Larger models get more GPUs for faster
# data throughput via DDP.
#
# Usage:
#   bash scripts/run_train/run_all_stage1a.sh               # run all 5 experiments
#   bash scripts/run_train/run_all_stage1a.sh --exp 06b_m64,4b_m128  # selected only
#
# Logs are written to: outputs/<exp_name>/stage1a_train.log
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

# ─── Parse arguments ────────────────────────────────────────────────
EXPERIMENTS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp) EXPERIMENTS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ─── Experiment definitions ─────────────────────────────────────────
# Format: name|gpus|label
ALL_EXPS=(
    "06b_m64|0|Qwen3-0.6B M=64"
    "06b_m128|1|Qwen3-0.6B M=128"
    "06b_m256|2|Qwen3-0.6B M=256"
    "17b_m128|3,4|Qwen3-1.7B M=128"
    "4b_m128|5,6,7|Qwen3-4B M=128"
)

# Map experiment name → script filename
declare -A SCRIPT_MAP=(
    ["06b_m64"]="stage1a_qwen06b_m64.sh"
    ["06b_m128"]="stage1a_qwen06b_m128.sh"
    ["06b_m256"]="stage1a_qwen06b_m256.sh"
    ["17b_m128"]="stage1a_qwen17b_m128.sh"
    ["4b_m128"]="stage1a_qwen4b_m128.sh"
)

# Map experiment name → output dir (for log files)
declare -A OUTPUT_MAP=(
    ["06b_m64"]="outputs/qwen06b_m64"
    ["06b_m128"]="outputs/qwen06b_m128"
    ["06b_m256"]="outputs/qwen06b_m256"
    ["17b_m128"]="outputs/qwen17b_m128"
    ["4b_m128"]="outputs/qwen4b_m128"
)

# ─── Filter experiments if --exp given ──────────────────────────────
if [[ -n "${EXPERIMENTS}" ]]; then
    IFS=',' read -ra FILTER <<< "${EXPERIMENTS}"
    SELECTED=()
    for entry in "${ALL_EXPS[@]}"; do
        name="${entry%%|*}"
        for f in "${FILTER[@]}"; do
            if [[ "${f}" == "${name}" ]]; then
                SELECTED+=("${entry}")
                break
            fi
        done
    done
    if [[ ${#SELECTED[@]} -eq 0 ]]; then
        echo "Error: no matching experiments. Available: 06b_m64 06b_m128 06b_m256 17b_m128 4b_m128"
        exit 1
    fi
    ALL_EXPS=("${SELECTED[@]}")
fi

# ─── Helper: count GPUs from comma-separated string ─────────────────
count_gpus() {
    local gpus="$1"
    echo "$gpus" | tr ',' '\n' | wc -l | xargs
}

# ─── Launch function ────────────────────────────────────────────────
launch_experiment() {
    local name="$1"
    local gpus="$2"
    local label="$3"
    local num_gpus
    num_gpus=$(count_gpus "${gpus}")
    local script="${SCRIPT_MAP[$name]}"
    local outdir="${OUTPUT_MAP[$name]}"

    mkdir -p "${outdir}"
    local logfile="${outdir}/stage1a_train.log"

    echo "  [LAUNCH] ${label} → GPU ${gpus} (${num_gpus} GPU(s)) | log: ${logfile}"

    CUDA_VISIBLE_DEVICES="${gpus}" NUM_GPUS="${num_gpus}" \
        bash "${SCRIPT_DIR}/${script}" \
        > "${logfile}" 2>&1 &
}

# ─── Main ───────────────────────────────────────────────────────────
echo "=========================================================="
echo "  QCPC Stage-1a Parallel Training (8x H200)"
echo "=========================================================="
echo ""
echo "  GPU allocation:"
for entry in "${ALL_EXPS[@]}"; do
    IFS='|' read -r name gpus label <<< "${entry}"
    num_gpus=$(count_gpus "${gpus}")
    printf "    GPU %-7s : %-25s (%d GPU)\n" "${gpus}" "${label}" "${num_gpus}"
done
echo ""
echo "  Launching ${#ALL_EXPS[@]} experiments in parallel..."
echo ""

PIDS=()
NAMES=()

for entry in "${ALL_EXPS[@]}"; do
    IFS='|' read -r name gpus label <<< "${entry}"
    launch_experiment "${name}" "${gpus}" "${label}"
    PIDS+=($!)
    NAMES+=("${name}")
done

echo ""
echo "  All experiments launched. Waiting for completion..."
echo ""

# ─── Wait and report ────────────────────────────────────────────────
FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    if wait "${pid}"; then
        echo "  [DONE] ${name} (PID ${pid}) ✓"
    else
        echo "  [FAIL] ${name} (PID ${pid}) ✗  — check ${OUTPUT_MAP[$name]}/stage1a_train.log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================================="
if [[ ${FAILED} -eq 0 ]]; then
    echo "  All ${#ALL_EXPS[@]} experiments completed successfully!"
else
    echo "  ${FAILED}/${#ALL_EXPS[@]} experiments FAILED. Check logs above."
fi
echo "=========================================================="

exit ${FAILED}
