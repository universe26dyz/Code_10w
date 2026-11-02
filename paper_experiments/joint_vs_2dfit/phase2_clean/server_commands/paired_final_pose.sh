#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: bash $0 PROPOSED_RUN FIT_FIRST_RUN ROI_BUNDLE SUBJECT [CASE ...]" >&2
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
EVAL_ROOT=$FIT_BASE/evaluation/final_pose/$SUBJECT
if [ "$#" -eq 0 ]; then CASES=(L100_fold0); else CASES=("$@"); fi

if ! command -v "$PY" >/dev/null 2>&1 || [ ! -d "$REPO" ] || [ ! -f "$NATIVE_MANIFEST" ] || [ ! -f "$ROI_BUNDLE/roi_derivation_manifest.json" ]; then
  echo "Required Python/repository/native SAX/ROI bundle is missing; no downstream work started." >&2
  exit 2
fi
cd "$REPO"

"$PY" - "$NATIVE_MANIFEST" <<'PY'
import json, pathlib, sys
from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first import require_scmr_v5_native_cache_manifest
path = pathlib.Path(sys.argv[1])
require_scmr_v5_native_cache_manifest(json.loads(path.read_text()), stack="SCMR-v5 SAX")
print("native-cache mapping contract: SCMR-v5 MATLAB PASS")
PY
PREP_SAX=$("$PY" -c '
import json,sys
value=json.load(open(sys.argv[1])).get("source_observations",{}).get("path")
if not isinstance(value,str) or not value: raise SystemExit("native SAX manifest lacks source_observations.path")
print(value)
' "$NATIVE_MANIFEST")
[ -f "$PREP_SAX" ] || { echo "Authoritative native-cache source observations are missing: $PREP_SAX" >&2; exit 1; }

run_new() {
  local label=$1 output=$2
  shift 2
  if [ -e "$output" ]; then
    [ -f "$output/manifest.json" ] || [ -f "$output" ] && { echo "SKIP existing $label: $output"; return 0; }
    echo "REFUSING partial existing output for $label: $output" >&2; return 1
  fi
  echo "START $label"; "$@"
}

mkdir -p "$EVAL_ROOT/bundles/proposed" "$EVAL_ROOT/bundles/fit_first" "$EVAL_ROOT/roi/mapping" "$EVAL_ROOT/metrics" "$EVAL_ROOT/visual_qc_full_heart"
for CASE in "${CASES[@]}"; do
  CASE_ROOT=$PROPOSED_BASE/sparse_inputs/$SUBJECT/$CASE
  MANIFEST=$CASE_ROOT/sparse_manifest.json
  PROPOSED=$PROPOSED_BASE/reconstruction/proposed/$SUBJECT/$CASE
  FIT=$FIT_BASE/reconstruction/fit_first/$SUBJECT/$CASE
  DIRECT_BUNDLE=$EVAL_ROOT/bundles/proposed/$CASE
  FIT_BUNDLE=$EVAL_ROOT/bundles/fit_first/$CASE
  MAPPING=$EVAL_ROOT/roi/mapping/$CASE.json
  METRICS=$EVAL_ROOT/metrics/$CASE
  AHA_SD_1PX=$METRICS/aha16_sd_myocardium_core_1px
  AHA_SD_LEGACY=$METRICS/aha16_sd_myocardium_core_legacy
  VISUAL_QC=$EVAL_ROOT/visual_qc_full_heart/$CASE
  [ -f "$MANIFEST" ] && [ -f "$PROPOSED/model.pt" ] && [ -f "$FIT/input_manifest.json" ] || { echo "REFUSING $CASE: sparse manifest or completed reconstruction is missing" >&2; exit 1; }
  run_new "$CASE Proposed final-pose bundle" "$DIRECT_BUNDLE" "$PY" -m paper_experiments.joint_vs_2dfit.aha.cyj11_r3_direct_final_pose_bundle --checkpoint "$PROPOSED/model.pt" --prepared-root "$CASE_ROOT" --sparse-manifest "$MANIFEST" --subject "$SUBJECT" --case "$CASE" --output "$DIRECT_BUNDLE" --device cuda:0
  run_new "$CASE Fit-first final-pose bundle" "$FIT_BUNDLE" "$PY" -m paper_experiments.joint_vs_2dfit.aha.current_fit_first_final_pose_bundle --baseline-root "$FIT" --native-maps-root "$NATIVE_ROOT" --subject "$SUBJECT" --case "$CASE" --sparse-manifest "$MANIFEST" --output "$FIT_BUNDLE"
  run_new "$CASE AHA mapping" "$MAPPING" "$PY" -m paper_experiments.joint_vs_2dfit.aha.derive_final_pose_mapping --subject "$SUBJECT" --case "$CASE" --prepared-sax "$PREP_SAX" --roi-bundle "$ROI_BUNDLE" --output "$MAPPING"
  run_new "$CASE paired AHA and myocardial metrics" "$METRICS" "$PY" -m paper_experiments.joint_vs_2dfit.aha.paired_final_pose_postprocess --subject "$SUBJECT" --case "$CASE" --native-sax-root "$NATIVE_ROOT/sax" --prepared-sax "$PREP_SAX" --mapping "$MAPPING" --roi-bundle "$ROI_BUNDLE" --outer-mask-bundle "$ROI_BUNDLE" --proposed-bundle "$DIRECT_BUNDLE" --fit-first-bundle "$FIT_BUNDLE" --output "$METRICS"
  run_new "$CASE 1px-core AHA SD bullseyes" "$AHA_SD_1PX" "$PY" -m paper_experiments.joint_vs_2dfit.aha.paired_aha_sd --segment-statistics-csv "$METRICS/aha16/aha16_segment_statistics_myocardium_core_1px.csv" --output "$AHA_SD_1PX"
  run_new "$CASE legacy-core AHA SD bullseyes" "$AHA_SD_LEGACY" "$PY" -m paper_experiments.joint_vs_2dfit.aha.paired_aha_sd --segment-statistics-csv "$METRICS/aha16/aha16_segment_statistics_myocardium_core_legacy.csv" --output "$AHA_SD_LEGACY"
  run_new "$CASE full-heart shared-mask through-plane and paired-slice QC" "$VISUAL_QC" "$PY" -m paper_experiments.joint_vs_2dfit.aha.paired_final_pose_through_plane --subject "$SUBJECT" --case "$CASE" --native-sax-root "$NATIVE_ROOT/sax" --prepared-sax "$PREP_SAX" --proposed-bundle "$DIRECT_BUNDLE" --fit-first-bundle "$FIT_BUNDLE" --proposed-t1 "$PROPOSED/T1_3D.nii.gz" --proposed-t2 "$PROPOSED/T2_3D.nii.gz" --fit-first-t1 "$FIT/nesvor/t1/volume_ms.nii.gz" --fit-first-t2 "$FIT/nesvor/t2/volume_ms.nii.gz" --output "$VISUAL_QC"
done
echo "paired final-pose downstream evaluation: PASS"
