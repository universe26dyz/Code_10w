#!/usr/bin/env bash
set -euo pipefail

CODE10W_ROOT=/data/dengyz/code/Code_10w
DATA_ROOT=/data/dengyz/dataset/Code_10w_v1
CROPPED_ROOT="${DATA_ROOT}/Code_10w_prepared"
B6_EXP="${DATA_ROOT}/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1"
B6_MODEL="${B6_EXP}/runs/frozen_mlp_B6/model.pt"
SIGNAL="${B6_EXP}/inputs/mlp/signal_simulator_approved.pth"
OUTPUT="${DATA_ROOT}/Code_10w_runs/quality_experiments_v1/CYJ/Q002S640_B6_roi_same_anchor_mse_v1"
LOG="${DATA_ROOT}/Code_10w_runs/quality_experiments_v1/CYJ/Q002S640_B6_roi_same_anchor_mse_v1.log"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cr_dreme
cd "${CODE10W_ROOT}"
[[ -z "$(git status --porcelain)" ]] || { echo "Refusing dirty Git worktree" >&2; exit 1; }
python -c 'import torch; assert torch.cuda.is_available()'
[[ -f "${B6_MODEL}" && -f "${SIGNAL}" ]] || { echo "Missing B6 model or approved decoder" >&2; exit 1; }
for stack in sax 2ch 4ch; do [[ -f "${CROPPED_ROOT}/CYJ/${stack}/observations.npz" ]] || { echo "Missing cropped observations for ${stack}" >&2; exit 1; }; done
[[ ! -e "${OUTPUT}" || -z "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || { echo "Refusing non-empty output: ${OUTPUT}" >&2; exit 1; }
[[ ! -e "${LOG}" ]] || { echo "Refusing existing log: ${LOG}" >&2; exit 1; }
sha256sum "${B6_MODEL}" "${SIGNAL}"
python -m quality_experiments.Q002_B6_roi_same_anchor_mse.run_q002_reconstruction --cropped-prepared-root "${CROPPED_ROOT}" --b6-model "${B6_MODEL}" --signal-simulator "${SIGNAL}" --output "${OUTPUT}" --device cuda:0 --experiment-id Q002S640_B6_roi_same_anchor_mse --anchor-batch-size 640 2>&1 | tee "${LOG}"
