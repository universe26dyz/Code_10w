#!/usr/bin/env bash
set -euo pipefail
CODE10W_ROOT=/data/dengyz/code/Code_10w; DATA_ROOT=/data/dengyz/dataset/Code_10w_v1
BASELINE_EXP_ROOT="${DATA_ROOT}/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1"
ROOT="${DATA_ROOT}/Code_10w_runs/quality_experiments_v1/CYJ/Q001B_full_fov_reg_cropped_recon_v1"
OUTPUT="${ROOT}/evaluation/map_domain_psf_K32"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cr_dreme; cd "${CODE10W_ROOT}"
[[ ! -e "${OUTPUT}" || -z "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || { echo "Refusing non-empty post-hoc output: ${OUTPUT}" >&2; exit 1; }
python -m quality_experiments.Q001_recon.run_map_domain_psf --mode Q001B --run-root "${ROOT}" --prepared-root "${DATA_ROOT}/Code_10w_prepared" --reference-root "${BASELINE_EXP_ROOT}/inputs/reference/native_reference_2d" --preprocessed-root "${BASELINE_EXP_ROOT}/inputs/preprocessed" --output "${OUTPUT}"
