#!/usr/bin/env bash
set -euo pipefail
# not auto-launched: an operator must invoke this formal server script with a new run id.
RUN_ID=${1:?run_id_required}; "$(dirname "$0")/server_preflight_phase2.sh" >/tmp/phase2_preflight.json
REPO=${REPO:-/data/dengyz/code/Code_10w}; DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}; cd "$REPO"
for SID in CYJ DYZ HHZ HJL; do OUTPUT="$DATA/Code_10w_runs/phase2_clean/$RUN_ID/$SID"; test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }; python -m paper_experiments.joint_vs_2dfit.phase2_clean.runner --method fit_first --dataset real_full --subject-id "$SID" --input-manifest paper_experiments/joint_vs_2dfit/phase2_clean/phase2_input_manifest.json --method-manifest paper_experiments/joint_vs_2dfit/phase2_clean/phase2_input_manifest.json; python -m paper_experiments.joint_vs_2dfit.phase2_clean.baseline --prepared-root "$DATA/Code_10w_prepared" --subject-id "$SID" --output "$OUTPUT" --execute 2>&1 | tee "$OUTPUT.log"; done
