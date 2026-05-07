#!/usr/bin/env bash
# Launch hamming_distance_plot.py for the 5 finished runs in sweep f1pktfh5.
# Distributes 5 runs across GPUs 4,5,6,7: first 4 in parallel, then 5th on GPU 4.
#
# Each run uses a distinct --pattern-seed so the 5 x 20 = 100 tested patterns
# per damage size are diverse across seeds.
#
# Output dirs are pre-computed (single shared SWEEP_TS) and recorded in
# ${LOG_DIR}/manifest.txt so downstream pooling can target this sweep's runs
# unambiguously, even when older sweeps share OUT_ROOT.
#
# Mirrors scripts/launch_accuracy_vs_damage_sweep.sh — same runs, same conditions,
# but produces hamming-distance summaries instead of accuracy-vs-damage.

set -euo pipefail

cd /home/marcello/workspace/boolean_nca_cc
source .venv/bin/activate

KO_SIZES="10,40,70,90,120,150,180,210"
PATTERNS_PER_SIZE=20
METHODS=both
WANDB_PROJECT=boolean_nca_cc  # sweep f1pktfh5 lives in the underscored project
FILENAME=latest_checkpoint  # use final training state, not first-time-at-1.0 best checkpoint

# (run_id, training_seed) — pattern_seed = 1000 + training_seed
RUN_IDS=(5a66oyt1 z434epjj uiyiujey brt24qbj 1oh9d7h9)
PAT_SEEDS=(1014    1123     1042     1456     1444)

# Where each run dumps its summary.csv. Single shared timestamp lets us pre-compute
# all 5 output paths and emit a manifest before launching any python process.
OUT_ROOT=reports/figures/hamming
SWEEP_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=logs/hamming_distance_sweep_${SWEEP_TS}
mkdir -p "${LOG_DIR}"
MANIFEST="${LOG_DIR}/manifest.txt"
echo "Logs: ${LOG_DIR}"

# Pre-compute output dirs (must match hamming_distance_plot.py's default naming
# template for include_all_gates=True, damage_behavior=reversible).
KO_PART="${KO_SIZES//,/_}"
OUT_DIRS=()
for run_id in "${RUN_IDS[@]}"; do
    OUT_DIRS+=("${OUT_ROOT}/hamming_scaled${KO_PART}_p${PATTERNS_PER_SIZE}_reversible_allgates_run${run_id}_${SWEEP_TS}")
done

# Write manifest upfront so downstream tooling can be pointed at it even if a
# run later fails (missing summary.csv will then be visible and explicit).
: > "${MANIFEST}"
for out_dir in "${OUT_DIRS[@]}"; do
    echo "${out_dir}" >> "${MANIFEST}"
done
echo "Manifest: ${MANIFEST}"

run_one() {
    local gpu=$1
    local run_id=$2
    local pattern_seed=$3
    local out_dir=$4
    local log_file="${LOG_DIR}/run_${run_id}.log"
    echo "[GPU ${gpu}] launching run_id=${run_id} pattern_seed=${pattern_seed} → ${log_file}"
    echo "                output_dir=${out_dir}"
    CUDA_VISIBLE_DEVICES=${gpu} \
    python experiments/hamming_distance_plot.py \
        --run-id "${run_id}" \
        --wandb-project "${WANDB_PROJECT}" \
        --filename "${FILENAME}" \
        --force-download \
        --methods "${METHODS}" \
        --knockout-sizes "${KO_SIZES}" \
        --patterns-per-size "${PATTERNS_PER_SIZE}" \
        --pattern-seed "${pattern_seed}" \
        --output "${out_dir}" \
        > "${log_file}" 2>&1
}

# First wave: 4 runs in parallel on GPUs 4,5,6,7.
GPUS=(4 5 6 7)
PIDS=()
for i in 0 1 2 3; do
    run_one "${GPUS[$i]}" "${RUN_IDS[$i]}" "${PAT_SEEDS[$i]}" "${OUT_DIRS[$i]}" &
    PIDS+=($!)
done

echo "Waiting on first wave (PIDs: ${PIDS[*]})..."
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        echo "PID ${pid} FAILED"
        FAILED=1
    fi
done

if [ "${FAILED}" -ne 0 ]; then
    echo "First wave had failures — check ${LOG_DIR}. Aborting before 5th run."
    exit 1
fi
echo "First wave done."

# Second wave: 5th run on GPU 4.
run_one 4 "${RUN_IDS[4]}" "${PAT_SEEDS[4]}" "${OUT_DIRS[4]}"

echo "All 5 runs complete. Pool with:"
echo "  python experiments/hamming_distance_REplot_multi.py \\"
echo "    --manifest ${MANIFEST} \\"
echo "    --output ${OUT_ROOT}/multi_seed_${SWEEP_TS}"
