#!/usr/bin/env bash
set -euo pipefail

CODE10W_ROOT=/data/dengyz/code/Code_10w
DATA_ROOT=/data/dengyz/dataset/Code_10w_v1
CROPPED_ROOT="${DATA_ROOT}/Code_10w_prepared"
B6_EXP="${DATA_ROOT}/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1"
B6_MODEL="${B6_EXP}/runs/frozen_mlp_B6/model.pt"
Q002_S640_ROOT="${DATA_ROOT}/Code_10w_runs/quality_experiments_v1/CYJ/Q002S640_B6_roi_same_anchor_mse_v1"
RUN_ROOT="${DATA_ROOT}/Code_10w_runs/quality_experiments_v1/CYJ/Q004S640_K1_B6_roi_same_anchor_mse_v1"
OUTPUT="${RUN_ROOT}/evaluation/formal_q004s640_k1_three_way_common_support"
LOG="${RUN_ROOT}/formal_q004s640_k1_three_way_common_support.log"
# The evaluator writes read-only comparator reprojections under ${OUTPUT}/derived_k1/.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cr_dreme
cd "${CODE10W_ROOT}"
[[ -z "$(git status --porcelain)" ]] || { echo "Refusing dirty Git worktree" >&2; exit 1; }
git rev-parse HEAD
[[ -f "${B6_MODEL}" && -f "${Q002_S640_ROOT}/model.pt" && -f "${Q002_S640_ROOT}/q002_route_manifest.json" && -f "${RUN_ROOT}/model.pt" && -f "${RUN_ROOT}/q002_route_manifest.json" ]] || { echo "Missing B6/Q002-S640/Q004 checkpoint or manifest" >&2; exit 1; }
python - "${Q002_S640_ROOT}" "${RUN_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
q002, q004 = (Path(value) for value in sys.argv[1:])
parent = json.loads((q002 / "q002_route_manifest.json").read_text())
if parent.get("experiment_id") != "Q002S640_B6_roi_same_anchor_mse": raise SystemExit("Q002-S640 parent manifest experiment_id mismatch")
if parent.get("q002_checkpoint_sha256") != hashlib.sha256((q002 / "model.pt").read_bytes()).hexdigest(): raise SystemExit("Q002-S640 parent checkpoint SHA256 mismatch")
route = json.loads((q004 / "q002_route_manifest.json").read_text())
if route.get("experiment_id") != "Q004S640_K1_B6_roi_same_anchor_mse" or route.get("q002_checkpoint_sha256") != hashlib.sha256((q004 / "model.pt").read_bytes()).hexdigest(): raise SystemExit("Q004 route manifest/checkpoint mismatch")
PY
[[ ! -e "${OUTPUT}" || -z "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || { echo "Refusing non-empty evaluation: ${OUTPUT}" >&2; exit 1; }
[[ ! -e "${LOG}" ]] || { echo "Refusing existing evaluator log: ${LOG}" >&2; exit 1; }
python -m quality_experiments.Q002_B6_roi_same_anchor_mse.evaluate_q004_s640_k1 --run-root "${RUN_ROOT}" --q002-s640-run-root "${Q002_S640_ROOT}" --b6-model "${B6_MODEL}" --b6-map-root "${B6_EXP}/runs/frozen_mlp_B6" --b6-d2-root "${DATA_ROOT}/Code_10w_runs/quality_experiments_v1/CYJ/D2_k8_signal_domain_baseline_v1" --cropped-prepared-root "${CROPPED_ROOT}" --reference-root "${B6_EXP}/inputs/reference/native_reference_2d" --preprocessed-root "${B6_EXP}/inputs/preprocessed" --output "${OUTPUT}" --device cuda:0 2>&1 | tee "${LOG}"
