#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/quality_experiments_v1/CYJ/Q001A_full_fov_full_recon_v1
[[ -f "${ROOT}/model.pt" && -f "${ROOT}/q001_route_manifest.json" ]] || { echo "Missing Q001A final checkpoint/route manifest" >&2; exit 1; }
CODE10W_ROOT=/data/dengyz/code/Code_10w; DATA_ROOT=/data/dengyz/dataset/Code_10w_v1; BASELINE_EXP_ROOT="${DATA_ROOT}/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cr_dreme; cd "${CODE10W_ROOT}"
python -m quality_experiments.Q001_recon.evaluate_q001 --mode Q001A --run-root "${ROOT}" --full-input-root "${DATA_ROOT}/Q001_full_fov_inputs_v2" --full-reference-root "${DATA_ROOT}/reference_data/CYJ/q001_full_fov_native_reference_v2" --cropped-reference-root "${BASELINE_EXP_ROOT}/inputs/reference/native_reference_2d" --output "${ROOT}/evaluation/formal_q001a"
