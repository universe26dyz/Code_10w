#!/usr/bin/env bash
# Build additive final-pose bundles for observed planes of one completed sparse case.
set -euo pipefail

ROOT=${FINAL5_ROOT:?set FINAL5_ROOT}
PY=${PHASE4R_PYTHON:-python}
SUBJECT=${SUBJECT:?set SUBJECT}
CASE=${CASE:?set CASE}
PREPARED_ROOT=${PREPARED_ROOT:?set PREPARED_ROOT exactly as declared by sparse_manifest.json}
DEVICE=${DEVICE:-cuda:0}

if [[ "$CASE" == "L100" ]]; then
  echo "L100 final-pose bundles are the existing formal bundle roots; no sparse bundle is built."
  exit 0
fi

CENTRAL="$ROOT/05_formal_endpoints/central"
SPARSE="$ROOT/03_sparse_inputs/$SUBJECT/$CASE/sparse_manifest.json"
NATIVE="$ROOT/02_standardized_12_15_10/native_dictionary/$SUBJECT"
PROPOSED_RUN="$ROOT/04_reconstruction/proposed/$SUBJECT/$CASE"
FIT_RUN="$ROOT/04_reconstruction/fit_first_exp1_1/$SUBJECT/$CASE"

"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.sparse_direct_native_bundles \
  --checkpoint "$PROPOSED_RUN/model.pt" --prepared-root "$PREPARED_ROOT" \
  --sparse-manifest "$SPARSE" --subject "$SUBJECT" --endpoint central \
  --output "$CENTRAL/proposed_native_bundle/$SUBJECT/$CASE" --device "$DEVICE"

"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_native_bundles \
  --baseline-root "$FIT_RUN" --r0-native-maps-root "$NATIVE" --endpoint central \
  --endpoint-source-root "$CENTRAL/fit_first/$SUBJECT/$CASE" \
  --output "$CENTRAL/fit_first_native_bundle/$SUBJECT/$CASE"
