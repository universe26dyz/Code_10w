#!/usr/bin/env bash
set -euo pipefail
# Server-only held-out map evaluation.  It does not construct signal-domain metrics.
[[ $# -eq 4 ]] || { echo "usage: $0 RUN_ID CONDITION PHASE NEW_OUTPUT" >&2; exit 2; }
RUN_ID=$1; CONDITION=$2; PHASE=$3; OUTPUT=$4; DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}; REPO=${REPO:-/data/dengyz/code/Code_10w}; PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
INPUT="$DATA/Code_10w_runs/paper_joint_vs_2dfit/sparse_inputs/CYJ/${CONDITION}_fold${PHASE}"; P="$DATA/Code_10w_runs/paper_joint_vs_2dfit/real/R1_sparse/$RUN_ID/proposed/CYJ/${CONDITION}_fold${PHASE}"; F="$DATA/Code_10w_runs/paper_joint_vs_2dfit/real/R1_sparse/$RUN_ID/fit_first/CYJ/${CONDITION}_fold${PHASE}"
test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }; cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.heldout --sparse-manifest "$INPUT/sparse_manifest.json" --proposed-t1 "$P/T1_3D.nii.gz" --proposed-t2 "$P/T2_3D.nii.gz" --fit-first-t1 "$F/nesvor/t1/volume_ms.nii.gz" --fit-first-t2 "$F/nesvor/t2/volume_ms.nii.gz" --output "$OUTPUT"
