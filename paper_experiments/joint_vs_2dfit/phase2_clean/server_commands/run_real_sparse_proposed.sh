#!/usr/bin/env bash
set -euo pipefail
# Server-only formal-grid runner; use the pilot wrapper for the CYJ pilot.
[[ $# -eq 4 ]] || { echo "usage: $0 RUN_ID SUBJECT CONDITION PHASE" >&2; exit 2; }
RUN_ID=$1; SID=$2; CONDITION=$3; PHASE=$4; test "$CONDITION" != L100 || { echo 'L100 reuses R0; no sparse reconstruction is allowed' >&2; exit 2; }
DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}; REPO=${REPO:-/data/dengyz/code/Code_10w}; PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}; INPUT="$DATA/Code_10w_runs/paper_joint_vs_2dfit/sparse_inputs/$SID/${CONDITION}_fold${PHASE}"; OUT="$DATA/Code_10w_runs/paper_joint_vs_2dfit/real/R1_sparse/$RUN_ID/proposed/$SID/${CONDITION}_fold${PHASE}"
test -f "$INPUT/sparse_manifest.json"; test ! -e "$OUT" || { echo "refusing overwrite: $OUT" >&2; exit 1; }; cd "$REPO"; DECODER=${PHASE2_APPROVED_DECODER:-$DATA/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/inputs/mlp/signal_simulator_approved.pth}; B6="$DATA/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/runs/frozen_mlp_B6/model.pt"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.proposed --prepared-root "$INPUT" --subject-id "$SID" --decoder "$DECODER" --b6-model "$B6" --output "$OUT" --execute
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.run_provenance --output "$OUT" --method proposed --subject "$SID" --prepared-root "$INPUT" --decoder "$DECODER" --b6-model "$B6"
