#!/usr/bin/env bash
# Read-only Final5 single-element publication renderer.  It never reconstructs or recomputes metrics.
set -euo pipefail

ROOT=${FINAL5_ROOT:-/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/phase4r_final_five_subject_k8_central_20261023}
PY=${PHASE4R_PYTHON:-python}
SUBJECT=${SUBJECT:?set SUBJECT}
CASE=${CASE:?set CASE}
OUTPUT=${OUTPUT:-"$ROOT/07_publication_visualization/single_element_v1/$SUBJECT/$CASE"}

exec "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.final5_publication_visualization \
  --final5-root "$ROOT" --subject "$SUBJECT" --case "$CASE" --output "$OUTPUT" "$@"
