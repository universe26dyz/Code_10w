#!/usr/bin/env bash
# Server-only auxiliary analysis: reads explicit completed maps and legacy masks.
set -euo pipefail

[[ $# -eq 2 ]] || { echo "usage: $0 COMPATIBILITY_CONFIG_JSON NEW_COMPATIBILITY_OUTPUT_ROOT" >&2; exit 2; }
REPO=${REPO:-/data/dengyz/code/Code_10w}
PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
CONFIG=$(realpath -m "$1")
OUTPUT=$(realpath -m "$2")
test -f "$CONFIG"
test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }

cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.scmr_compatibility_eval --config "$CONFIG" --output "$OUTPUT"
