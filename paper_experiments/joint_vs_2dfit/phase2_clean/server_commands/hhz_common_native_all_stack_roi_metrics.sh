#!/usr/bin/env bash
# Evaluate the stacked manual SAX LV/myocardial mask volume on all common-native stacks.

if [ "$#" -lt 4 ]; then
  echo "Usage: bash $0 /data/.../proposed_run /data/.../fit_first_run /path/to/hhz_roi_bundle L100_fold0 [other cases...]"
  exit 2
fi

PROPOSED_BASE=$1
FIT_BASE=$2
ROI_BUNDLE=$3
shift 3
REPO=${REPO:-/data/dengyz/code/Code_10w}
PY=${PY:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
SUBJECT=HHZ
NATIVE_ROOT=$FIT_BASE/reconstruction/fit_first/$SUBJECT/L100_fold0/native_maps
NATIVE_MANIFEST=$NATIVE_ROOT/sax/manifest.json
OUT_ROOT=$FIT_BASE/evaluation/common_native/$SUBJECT

if [ ! -x "$PY" ] || [ ! -d "$REPO" ] || [ ! -f "$NATIVE_MANIFEST" ] || [ ! -f "$ROI_BUNDLE/sax_myocardium_masks.npz" ]; then
  echo "Required Python/repository/native cache/ROI archive is missing; no work started."
  exit 2
fi

cd "$REPO" || exit 2
PREP_SAX=$($PY -c '
import json,sys
value=json.load(open(sys.argv[1])).get("source_observations",{}).get("path")
if not isinstance(value,str) or not value: raise SystemExit("native SAX manifest lacks source_observations.path")
print(value)
' "$NATIVE_MANIFEST")
PREP_SUBJECT=$(dirname "$(dirname "$PREP_SAX")")

for CASE in "$@"; do
  PROPOSED=$PROPOSED_BASE/reconstruction/proposed/$SUBJECT/$CASE
  FIT=$FIT_BASE/reconstruction/fit_first/$SUBJECT/$CASE
  SPARSE_MANIFEST=$PROPOSED_BASE/sparse_inputs/$SUBJECT/$CASE/sparse_manifest.json
  COMMON_CASE=$OUT_ROOT/$CASE
  OUTPUT=$COMMON_CASE/all_stack_roi_metrics
  if [ ! -f "$COMMON_CASE/manifest.json" ]; then
    echo "REFUSING $CASE: completed common-native output is missing: $COMMON_CASE"
    exit 1
  fi
  if [ -e "$OUTPUT" ]; then
    echo "REFUSING existing all-stack ROI output: $OUTPUT"
    exit 1
  fi
  if [ ! -f "$SPARSE_MANIFEST" ]; then
    echo "REFUSING $CASE: sparse manifest is missing: $SPARSE_MANIFEST"
    exit 1
  fi
  for PATH_ITEM in \
    "$PROPOSED/T1_3D.nii.gz" "$PROPOSED/T2_3D.nii.gz" \
    "$FIT/nesvor/t1/volume_ms.nii.gz" "$FIT/nesvor/t2/volume_ms.nii.gz"; do
    if [ ! -f "$PATH_ITEM" ]; then
      echo "REFUSING $CASE: missing completed volume $PATH_ITEM"
      exit 1
    fi
  done
  echo "START $CASE all-stack common-native LV/myocardial ROI metrics"
  "$PY" -m paper_experiments.joint_vs_2dfit.aha.common_native_all_stack_roi_metrics \
    --subject "$SUBJECT" --case "$CASE" --native-maps-root "$NATIVE_ROOT" \
    --prepared-subject-root "$PREP_SUBJECT" --roi-bundle "$ROI_BUNDLE" \
    --sparse-manifest "$SPARSE_MANIFEST" \
    --proposed-t1 "$PROPOSED/T1_3D.nii.gz" --proposed-t2 "$PROPOSED/T2_3D.nii.gz" \
    --fit-first-t1 "$FIT/nesvor/t1/volume_ms.nii.gz" --fit-first-t2 "$FIT/nesvor/t2/volume_ms.nii.gz" \
    --output "$OUTPUT" || exit 1
done

echo "HHZ all-stack common-native LV/myocardial ROI metrics: PASS"
