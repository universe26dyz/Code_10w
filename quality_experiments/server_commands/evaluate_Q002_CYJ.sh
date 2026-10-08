#!/usr/bin/env bash
set -euo pipefail

CODE10W_ROOT=/data/dengyz/code/Code_10w
DATA_ROOT=/data/dengyz/dataset/Code_10w_v1
B6_EXP="${DATA_ROOT}/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1"
RUN_ROOT="${DATA_ROOT}/Code_10w_runs/quality_experiments_v1/CYJ/Q002_B6_roi_same_anchor_mse_v2"
OUTPUT="${RUN_ROOT}/evaluation/formal_q002_b6_roi_common_support"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cr_dreme
cd "${CODE10W_ROOT}"
[[ -z "$(git status --porcelain)" ]] || { echo "Refusing dirty Git worktree" >&2; exit 1; }
[[ -f "${RUN_ROOT}/model.pt" && -f "${RUN_ROOT}/q002_route_manifest.json" ]] || { echo "Missing Q002 checkpoint/manifest" >&2; exit 1; }
[[ ! -e "${OUTPUT}" || -z "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || { echo "Refusing non-empty evaluation: ${OUTPUT}" >&2; exit 1; }
python -m quality_experiments.Q002_B6_roi_same_anchor_mse.evaluate_q002 --run-root "${RUN_ROOT}" --b6-map-root "${B6_EXP}/runs/frozen_mlp_B6" --b6-d2-root "${DATA_ROOT}/Code_10w_runs/quality_experiments_v1/CYJ/D2_k8_signal_domain_baseline_v1" --reference-root "${B6_EXP}/inputs/reference/native_reference_2d" --preprocessed-root "${B6_EXP}/inputs/preprocessed" --output "${OUTPUT}"
