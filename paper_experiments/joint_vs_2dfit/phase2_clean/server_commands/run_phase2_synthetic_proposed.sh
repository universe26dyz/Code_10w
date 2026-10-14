#!/usr/bin/env bash
set -euo pipefail
# not auto-launched: an operator must invoke this formal server script with a new run id.
RUN_ID=${1:?run_id_required}; CR_PYTHON=${PHASE2_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}; test -x "$CR_PYTHON"; "$(dirname "$0")/server_preflight_phase2.sh" >/tmp/phase2_preflight.json
REPO=${REPO:-/data/dengyz/code/Code_10w}; DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}; DECODER=$("$CR_PYTHON" -c 'import json; print(json.load(open("/tmp/phase2_preflight.json"))["approved_decoder"]["path"])'); OUTPUT="$DATA/Code_10w_runs/phase2_clean/$RUN_ID"
test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }; cd "$REPO"
"$CR_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.proposed --prepared-root "$REPO/paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_1_clean_mrxcat" --subject-id synthetic_clean --decoder "$DECODER" --b6-model "$DATA/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/runs/frozen_mlp_B6/model.pt" --output "$OUTPUT" --execute 2>&1 | tee "$OUTPUT.log"
