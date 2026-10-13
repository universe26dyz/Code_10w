#!/usr/bin/env bash
set -euo pipefail
# not auto-launched: an operator must invoke this formal server script with a new run id.
RUN_ID=${1:?run_id_required}; "$(dirname "$0")/server_preflight_phase2.sh" >/tmp/phase2_preflight.json
REPO=${REPO:-/data/dengyz/code/Code_10w}; DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}; NESVOR_EXEC=$(python -c 'import json; print(json.load(open("/tmp/phase2_preflight.json"))["historical_nesvor"]["executable"])'); cd "$REPO"
for SID in CYJ DYZ HHZ HJL; do OUTPUT="$DATA/Code_10w_runs/phase2_clean/$RUN_ID/$SID"; test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }; python -m paper_experiments.joint_vs_2dfit.phase2_clean.baseline --prepared-root "$DATA/Code_10w_prepared" --subject-id "$SID" --nesvor-executable "$NESVOR_EXEC" --output "$OUTPUT" --execute 2>&1 | tee "$OUTPUT.log"; done
