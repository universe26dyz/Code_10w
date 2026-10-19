#!/usr/bin/env bash
# Server-only deterministic aggregation of existing formal matched endpoint results.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 FORMAL_MATCHED_RESULTS_ROOT NEW_OUTPUT" >&2; exit 2; }
INPUT=$1; OUTPUT=$2
INPUT=$(realpath -m "$INPUT"); OUTPUT=$(realpath -m "$OUTPUT")
REPO=${REPO:-/data/dengyz/code/Code_10w}
PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
test -d "$INPUT"
test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
printf 'CYJ matched aggregation:\n  input=%s\n  output=%s\n' "$INPUT" "$OUTPUT"
cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj_matched_comparison aggregate --input-root "$INPUT" --output "$OUTPUT"
