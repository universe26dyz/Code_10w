#!/usr/bin/env bash
set -euo pipefail
# L100 is a full-data LAX agreement point: it reuses R0 outputs and never reconstructs again.
[[ $# -eq 3 ]] || { echo "usage: $0 R0_RUN_ID SUBJECT NEW_OUTPUT" >&2; exit 2; }
RUN_ID=$1; SID=$2; OUTPUT=$3; DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}; REPO=${REPO:-/data/dengyz/code/Code_10w}; PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}; INPUT="$DATA/Code_10w_runs/paper_joint_vs_2dfit/sparse_inputs/$SID/L100_fold0/sparse_manifest.json"; R0="$DATA/Code_10w_runs/paper_joint_vs_2dfit/real/R0_full/$RUN_ID"
test -f "$INPUT"; test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }; cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.heldout --sparse-manifest "$INPUT" --proposed-t1 "$R0/proposed/$SID/T1_3D.nii.gz" --proposed-t2 "$R0/proposed/$SID/T2_3D.nii.gz" --fit-first-t1 "$R0/fit_first/$SID/nesvor/t1/volume_ms.nii.gz" --fit-first-t2 "$R0/fit_first/$SID/nesvor/t2/volume_ms.nii.gz" --output "$OUTPUT"
