#!/usr/bin/env bash
# Read-only post-hoc L100-sampling-mask evaluation; no training/reconstruction/refit.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 MASK_AUDIT_ROOT NEW_OUTPUT" >&2; exit 2; }
AUDIT=$1
OUTPUT=$2
REPO=${REPO:-/data/dengyz/code/Code_10w}
DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}
PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
R0_RUN_ID=${R0_RUN_ID:-R0_PHASE4R}
R1_RUN_ID=${R1_RUN_ID:-R1_PHASE4R_PILOT1}
SUBJECT=CYJ
CONDITION=L25_fold0
SPARSE_MANIFEST="$DATA/Code_10w_runs/paper_joint_vs_2dfit/sparse_inputs/$SUBJECT/$CONDITION/sparse_manifest.json"
R0_NATIVE="$DATA/Code_10w_runs/paper_joint_vs_2dfit/real/R0_native_cache/$R0_RUN_ID/$SUBJECT/native_maps"
R1_REAL="$DATA/Code_10w_runs/paper_joint_vs_2dfit/real/R1_sparse"
test -d "$AUDIT" && test -f "$SPARSE_MANIFEST" && test -d "$R0_NATIVE"
test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.l100_mask_paired_eval \
  --subject "$SUBJECT" --condition "$CONDITION" --r0-run-id "$R0_RUN_ID" --r1-run-id "$R1_RUN_ID" \
  --sparse-manifest "$SPARSE_MANIFEST" --r0-native-maps-root "$R0_NATIVE" \
  --r1-real-root "$R1_REAL" --mask-audit-root "$AUDIT" --output "$OUTPUT"
