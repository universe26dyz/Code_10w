#!/usr/bin/env bash
set -euo pipefail
# not auto-launched: an operator must invoke this formal server script with a new run id.
RUN_ID=${1:?run_id_required}; CR_PYTHON=${PHASE2_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}; test -x "$CR_PYTHON"; "$(dirname "$0")/server_preflight_phase2.sh" >/tmp/phase2_preflight.json
REPO=${REPO:-/data/dengyz/code/Code_10w}; DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}; NESVOR_EXEC=$("$CR_PYTHON" -c 'import json; print(json.load(open("/tmp/phase2_preflight.json"))["historical_nesvor"]["executable"])'); SYNTHETIC_ROOT=${PHASE1_2_SYNTHETIC_ROOT:-"$REPO/paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_2_xcat3d_clean"}; OUTPUT="$DATA/Code_10w_runs/phase2_clean/$RUN_ID"
test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }; cd "$REPO"; "$CR_PYTHON" -m paper_experiments.joint_vs_2dfit.xcat3d.validate_quantitative_gt --manifest "$SYNTHETIC_ROOT/synthetic_source_manifest.json" >/dev/null; "$CR_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.baseline --prepared-root "$SYNTHETIC_ROOT" --subject-id synthetic_clean --nesvor-executable "$NESVOR_EXEC" --output "$OUTPUT" --execute 2>&1 | tee "$OUTPUT.log"
