#!/usr/bin/env bash
set -euo pipefail

CODE10W_ROOT=/data/dengyz/code/Code_10w
DATA_ROOT=/data/dengyz/dataset/Code_10w_v1
B6_EXP="${DATA_ROOT}/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1"
Q002_S640_ROOT="${DATA_ROOT}/Code_10w_runs/quality_experiments_v1/CYJ/Q002S640_B6_roi_same_anchor_mse_v1"
RUN_ROOT="${DATA_ROOT}/Code_10w_runs/quality_experiments_v1/CYJ/Q003S640_B6_roi_same_anchor_mse_plus_cosine_10k_v1"
OUTPUT="${RUN_ROOT}/evaluation/formal_q003s640_10k_four_way_common_support"
LOG="${RUN_ROOT}/formal_q003s640_10k_four_way_common_support.log"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cr_dreme
cd "${CODE10W_ROOT}"
[[ -z "$(git status --porcelain)" ]] || { echo "Refusing dirty Git worktree" >&2; exit 1; }
git rev-parse HEAD
[[ -f "${Q002_S640_ROOT}/model.pt" && -f "${Q002_S640_ROOT}/q002_route_manifest.json" ]] || { echo "Missing Q002-S640 parent checkpoint/manifest" >&2; exit 1; }
[[ -f "${RUN_ROOT}/model.pt" && -f "${RUN_ROOT}/q002_route_manifest.json" && -f "${RUN_ROOT}/checkpoints/model_iter_6000.pt" ]] || { echo "Missing Q003-10k checkpoint or manifest" >&2; exit 1; }
python - "${Q002_S640_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "q002_route_manifest.json").read_text())
if manifest.get("experiment_id") != "Q002S640_B6_roi_same_anchor_mse":
    raise SystemExit("Q002-S640 parent manifest experiment_id mismatch")
expected = manifest.get("q002_checkpoint_sha256") or manifest.get("checkpoint_sha256")
actual = hashlib.sha256((root / "model.pt").read_bytes()).hexdigest()
if expected != actual:
    raise SystemExit("Q002-S640 parent checkpoint SHA256 mismatch")
PY
[[ ! -e "${OUTPUT}" || -z "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || { echo "Refusing non-empty evaluation: ${OUTPUT}" >&2; exit 1; }
[[ ! -e "${LOG}" ]] || { echo "Refusing existing log: ${LOG}" >&2; exit 1; }
python -m quality_experiments.Q002_B6_roi_same_anchor_mse.evaluate_q003_s640_10k --run-root "${RUN_ROOT}" --q002-s640-run-root "${Q002_S640_ROOT}" --b6-map-root "${B6_EXP}/runs/frozen_mlp_B6" --b6-d2-root "${DATA_ROOT}/Code_10w_runs/quality_experiments_v1/CYJ/D2_k8_signal_domain_baseline_v1" --reference-root "${B6_EXP}/inputs/reference/native_reference_2d" --preprocessed-root "${B6_EXP}/inputs/preprocessed" --output "${OUTPUT}" 2>&1 | tee "${LOG}"
