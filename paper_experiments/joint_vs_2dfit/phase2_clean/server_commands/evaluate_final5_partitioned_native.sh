#!/usr/bin/env bash
# Additive Final5 role-partitioned native metrics; never overwrites formal metrics.
set -euo pipefail

ROOT=${FINAL5_ROOT:?set FINAL5_ROOT}
PY=${PHASE4R_PYTHON:-python}
SUBJECT=${SUBJECT:?set SUBJECT}
CASE=${CASE:?set CASE}

CENTRAL="$ROOT/05_formal_endpoints/central"
NATIVE="$ROOT/02_standardized_12_15_10/native_dictionary/$SUBJECT"
SPARSE="$ROOT/03_sparse_inputs/$SUBJECT/$CASE/sparse_manifest.json"
MASK="$ROOT/06_metrics/current_native_mask/$SUBJECT/L100"
OUTPUT=${OUTPUT:-"$ROOT/06_metrics/final_native_partitioned/$SUBJECT/$CASE"}

exec "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.final5_partitioned_native_metrics \
  --proposed-endpoint-root "$CENTRAL/proposed/$SUBJECT/$CASE" \
  --fit-first-endpoint-root "$CENTRAL/fit_first/$SUBJECT/$CASE" \
  --proposed-native-bundle-root "$CENTRAL/proposed_native_bundle/$SUBJECT/$CASE" \
  --fit-first-native-bundle-root "$CENTRAL/fit_first_native_bundle/$SUBJECT/$CASE" \
  --native-mask-root "$MASK" --r0-native-maps-root "$NATIVE" \
  --sparse-manifest "$SPARSE" --output "$OUTPUT"
