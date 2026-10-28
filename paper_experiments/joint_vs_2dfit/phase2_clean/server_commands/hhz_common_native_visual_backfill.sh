#!/usr/bin/env bash
# Add only missing common-native visual outputs; never reruns quantitative metrics or reconstruction.

if [ "$#" -lt 5 ]; then
  echo "Usage: bash $0 through-plane|aha /data/.../proposed_run /data/.../fit_first_run /path/to/hhz_roi_bundle L100_fold0 [other cases...]"
  exit 2
fi

MODE=$1
PROPOSED_BASE=$2
FIT_BASE=$3
ROI_BUNDLE=$4
shift 4
if [ "$MODE" = "through-plane" ]; then
  FLAG=--through-plane
elif [ "$MODE" = "aha" ]; then
  FLAG=--aha
else
  echo "Mode must be through-plane or aha"
  exit 2
fi

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
  OUTPUT=$OUT_ROOT/$CASE
  if [ ! -f "$OUTPUT/manifest.json" ]; then
    echo "REFUSING $CASE: completed common-native metrics output is missing: $OUTPUT"
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
  echo "START $CASE common-native $MODE visual backfill"
  "$PY" -m paper_experiments.joint_vs_2dfit.aha.common_native_visual_backfill \
    --subject "$SUBJECT" --case "$CASE" --native-maps-root "$NATIVE_ROOT" \
    --prepared-subject-root "$PREP_SUBJECT" --roi-bundle "$ROI_BUNDLE" \
    --proposed-t1 "$PROPOSED/T1_3D.nii.gz" --proposed-t2 "$PROPOSED/T2_3D.nii.gz" \
    --fit-first-t1 "$FIT/nesvor/t1/volume_ms.nii.gz" --fit-first-t2 "$FIT/nesvor/t2/volume_ms.nii.gz" \
    --output "$OUTPUT" "$FLAG" || exit 1
done

echo "HHZ common-native $MODE visual backfill: PASS"
