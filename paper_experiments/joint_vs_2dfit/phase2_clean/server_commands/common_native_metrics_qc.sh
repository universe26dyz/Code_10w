#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "Usage: bash $0 PROPOSED_RUN FIT_FIRST_RUN ROI_BUNDLE SUBJECT CASE [CASE ...]" >&2
  exit 2
fi

PROPOSED_BASE=$1
FIT_BASE=$2
ROI_BUNDLE=$3
SUBJECT=$4
shift 4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}
PY=${PY:-python}
NATIVE_ROOT=$FIT_BASE/reconstruction/fit_first/$SUBJECT/L100_fold0/native_maps
NATIVE_MANIFEST=$NATIVE_ROOT/sax/manifest.json
OUT_ROOT=${OUT_ROOT:-$FIT_BASE/evaluation/common_native/$SUBJECT}

if ! command -v "$PY" >/dev/null 2>&1 || [ ! -d "$REPO" ] || [ ! -f "$NATIVE_MANIFEST" ] || [ ! -f "$ROI_BUNDLE/sax_myocardium_masks.npz" ]; then
  echo "Required Python/repository/native cache/ROI archive is missing; no work started." >&2
  exit 2
fi
cd "$REPO"
PREP_SAX=$("$PY" -c '
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
  [ ! -e "$OUTPUT" ] || { echo "REFUSING existing corrected output: $OUTPUT" >&2; exit 1; }
  for PATH_ITEM in "$PROPOSED/T1_3D.nii.gz" "$PROPOSED/T2_3D.nii.gz" "$FIT/nesvor/t1/volume_ms.nii.gz" "$FIT/nesvor/t2/volume_ms.nii.gz"; do
    [ -f "$PATH_ITEM" ] || { echo "REFUSING $CASE: missing completed volume $PATH_ITEM" >&2; exit 1; }
  done
  echo "START $CASE common-native paired metrics and QC"
  "$PY" -m paper_experiments.joint_vs_2dfit.aha.common_native_metrics_qc \
    --subject "$SUBJECT" --case "$CASE" --native-maps-root "$NATIVE_ROOT" \
    --prepared-subject-root "$PREP_SUBJECT" --roi-bundle "$ROI_BUNDLE" \
    --proposed-t1 "$PROPOSED/T1_3D.nii.gz" --proposed-t2 "$PROPOSED/T2_3D.nii.gz" \
    --fit-first-t1 "$FIT/nesvor/t1/volume_ms.nii.gz" --fit-first-t2 "$FIT/nesvor/t2/volume_ms.nii.gz" \
    --output "$OUTPUT"
done
echo "common-native paired evaluation: PASS"
