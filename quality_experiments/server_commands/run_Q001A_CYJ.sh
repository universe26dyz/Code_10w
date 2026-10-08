#!/usr/bin/env bash
set -euo pipefail
CODE10W_ROOT=/data/dengyz/code/Code_10w; DATA_ROOT=/data/dengyz/dataset/Code_10w_v1; INPUT_ROOT="${DATA_ROOT}/Q001_full_fov_inputs_v2"; CROPPED_ROOT="${DATA_ROOT}/Code_10w_prepared"; B6_MODEL="${DATA_ROOT}/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/runs/frozen_mlp_B6/model.pt"; SIGNAL="${DATA_ROOT}/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/inputs/mlp/signal_simulator_approved.pth"; OUTPUT="${DATA_ROOT}/Code_10w_runs/quality_experiments_v1/CYJ/Q001A_full_fov_full_recon_v1"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cr_dreme; cd "${CODE10W_ROOT}"; python -c 'import torch; assert torch.cuda.is_available()'; Q001_INPUT_ROOT="${INPUT_ROOT}" bash quality_experiments/server_commands/verify_Q001_inputs_CYJ.sh
[[ -z "$(git status --porcelain)" ]] || { echo "Refusing formal Q001A run from dirty or untracked Git worktree" >&2; exit 1; }
[[ ! -e "${OUTPUT}.log" ]] || { echo "Refusing to overwrite existing Q001A log: ${OUTPUT}.log" >&2; exit 1; }
[[ ! -e "${OUTPUT}" || -z "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || { echo "Refusing non-empty output: ${OUTPUT}" >&2; exit 1; }
python -m quality_experiments.Q001_recon.run_controlled_reconstruction --experiment-id Q001A_full_fov_full_recon --full-input-root "${INPUT_ROOT}" --cropped-prepared-root "${CROPPED_ROOT}" --b6-model "${B6_MODEL}" --signal-simulator "${SIGNAL}" --output "${OUTPUT}" --device cuda:0 2>&1 | tee "${OUTPUT}.log"
