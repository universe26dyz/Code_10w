#!/usr/bin/env bash
set -euo pipefail
# Server-only formal-grid held-out quantitative evaluation.
[[ $# -eq 5 ]] || { echo "usage: $0 RUN_ID SUBJECT CONDITION PHASE NEW_OUTPUT" >&2; exit 2; }
RUN_ID=$1; SID=$2; CONDITION=$3; PHASE=$4; OUTPUT=$5; DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}; REPO=${REPO:-/data/dengyz/code/Code_10w}; PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}; INPUT="$DATA/Code_10w_runs/paper_joint_vs_2dfit/sparse_inputs/$SID/${CONDITION}_fold${PHASE}"; P="$DATA/Code_10w_runs/paper_joint_vs_2dfit/real/R1_sparse/$RUN_ID/proposed/$SID/${CONDITION}_fold${PHASE}"; F="$DATA/Code_10w_runs/paper_joint_vs_2dfit/real/R1_sparse/$RUN_ID/fit_first/$SID/${CONDITION}_fold${PHASE}"
test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }; cd "$REPO"
source "$REPO/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/r0_native_cache_path.sh"
CACHE=$(phase4r_resolve_r0_native_maps "$SID" "$RUN_ID"); printf 'R0 native cache: %s\n' "$CACHE" >&2
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.heldout --sparse-manifest "$INPUT/sparse_manifest.json" --proposed-t1 "$P/T1_3D.nii.gz" --proposed-t2 "$P/T2_3D.nii.gz" --fit-first-t1 "$F/nesvor/t1/volume_ms.nii.gz" --fit-first-t2 "$F/nesvor/t2/volume_ms.nii.gz" --r0-native-maps-root "$CACHE" --output "$OUTPUT"
