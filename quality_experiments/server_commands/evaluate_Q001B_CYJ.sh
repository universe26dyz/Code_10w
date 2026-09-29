#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/quality_experiments_v1/CYJ/Q001B_full_fov_reg_cropped_recon_v1
[[ -f "${ROOT}/model.pt" && -f "${ROOT}/q001_route_manifest.json" ]] || { echo "Missing Q001B final checkpoint/route manifest" >&2; exit 1; }
CODE10W_ROOT=/data/dengyz/code/Code_10w; DATA_ROOT=/data/dengyz/dataset/Code_10w_v1; BASELINE_EXP_ROOT="${DATA_ROOT}/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1"; CROPPED_PREPROCESSED_ROOT="${BASELINE_EXP_ROOT}/inputs/preprocessed"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cr_dreme; cd "${CODE10W_ROOT}"
python -m quality_experiments.Q001_recon.evaluate_q001 --mode Q001B --run-root "${ROOT}" --full-input-root "${DATA_ROOT}/Q001_full_fov_inputs_v2" --cropped-reference-root "${BASELINE_EXP_ROOT}/inputs/reference/native_reference_2d" --cropped-preprocessed-root "${CROPPED_PREPROCESSED_ROOT}" --b6-map-root "${BASELINE_EXP_ROOT}/runs/frozen_mlp_B6" --b6-d2-root "${DATA_ROOT}/Code_10w_runs/quality_experiments_v1/CYJ/D2_k8_signal_domain_baseline_v1" --output "${ROOT}/evaluation/formal_q001b"
