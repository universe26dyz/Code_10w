#!/usr/bin/env bash
set -euo pipefail
# Server-only formal-grid runner; use the pilot wrapper for the CYJ pilot.
[[ $# -eq 4 ]] || { echo "usage: $0 RUN_ID SUBJECT CONDITION PHASE" >&2; exit 2; }
RUN_ID=$1; SID=$2; CONDITION=$3; PHASE=$4; test "$CONDITION" != L100 || { echo 'L100 reuses R0; no sparse reconstruction is allowed' >&2; exit 2; }
DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}; REPO=${REPO:-/data/dengyz/code/Code_10w}; PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}; INPUT="$DATA/Code_10w_runs/paper_joint_vs_2dfit/sparse_inputs/$SID/${CONDITION}_fold${PHASE}"; OUT="$DATA/Code_10w_runs/paper_joint_vs_2dfit/real/R1_sparse/$RUN_ID/fit_first/$SID/${CONDITION}_fold${PHASE}"; R0_RUN_ID=${PHASE4R_R0_RUN_ID:-$RUN_ID}; R0_NATIVE=${PHASE4R_R0_NATIVE_MAPS_ROOT:-"$DATA/Code_10w_runs/paper_joint_vs_2dfit/real/R0_full/$R0_RUN_ID/fit_first/$SID/native_maps"}
test -f "$INPUT/sparse_manifest.json"; test -d "$R0_NATIVE"; test ! -e "$OUT" || { echo "refusing overwrite: $OUT" >&2; exit 1; }; cd "$REPO"; NESVOR=/home/dengyz/miniconda3/envs/nesvor/bin/nesvor
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.baseline --prepared-root "$INPUT" --subject-id "$SID" --reuse-native-maps-root "$R0_NATIVE" --sparse-manifest "$INPUT/sparse_manifest.json" --nesvor-executable "$NESVOR" --output "$OUT" --execute
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.run_provenance --output "$OUT" --method fit_first --subject "$SID" --prepared-root "$INPUT" --nesvor-executable "$NESVOR"
