#!/usr/bin/env bash
set -euo pipefail
# Server-only renderer for an existing R1 or L100 held-out evaluation directory; never reconstructs or refits.
[[ $# -eq 8 ]] || { echo "usage: $0 HELDOUT_ROOT STACK PROPOSED_T1 PROPOSED_T2 FIT_FIRST_T1 FIT_FIRST_T2 NEW_FIGURE_ROOT GROUP_OR_AUTO" >&2; exit 2; }
HELDOUT=$1; STACK=$2; PROPOSED_T1=$3; PROPOSED_T2=$4; FIT_T1=$5; FIT_T2=$6; OUTPUT=$7; GROUP=$8
REPO=${REPO:-/data/dengyz/code/Code_10w}; PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
test -f "$HELDOUT/heldout_quantitative_metrics.json"
test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
cd "$REPO"
ARGS=(--mode r1 --heldout-root "$HELDOUT" --stack "$STACK" --proposed-t1 "$PROPOSED_T1" --proposed-t2 "$PROPOSED_T2" --fit-first-t1 "$FIT_T1" --fit-first-t2 "$FIT_T2" --output "$OUTPUT" --subject heldout)
if [[ "$GROUP" != "auto" ]]; then ARGS+=(--group "$GROUP"); fi
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.visualization "${ARGS[@]}"
