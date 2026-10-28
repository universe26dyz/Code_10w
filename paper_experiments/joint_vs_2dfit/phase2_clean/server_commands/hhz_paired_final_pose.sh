#!/usr/bin/env bash
# Completed-HHZ downstream only: final-pose bundles, paired AHA/ROI, and
# shared-mask through-plane QC.  It never trains or reconstructs.

if [ "$#" -ne 2 ]; then
  echo "Usage: bash $0 /data/.../ismrm_v2 /path/to/verified_hhz_native_roi_bundle"
  exit 2
fi

BASE=$1
ROI_BUNDLE=$2
REPO=${REPO:-/data/dengyz/code/Code_10w}
PY=${PY:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
SUBJECT=HHZ
NATIVE_ROOT=$BASE/reconstruction/fit_first/$SUBJECT/L100_fold0/native_maps
NATIVE_MANIFEST=$NATIVE_ROOT/sax/manifest.json
EVAL_ROOT=$BASE/evaluation/final_pose/$SUBJECT
CASES=(L100_fold0 L50_fold0 L25_fold0 SAX100_2CH0_4CH0 SAX100_2CH0_4CH100 SAX100_2CH100_4CH0 S2CH100_4CH50_fold0 S2CH50_4CH100_fold0)

if [ ! -x "$PY" ] || [ ! -d "$REPO" ] || [ ! -f "$NATIVE_MANIFEST" ] || [ ! -f "$ROI_BUNDLE/roi_derivation_manifest.json" ]; then
  echo "Required Python/repository/native SAX/ROI bundle is missing; no downstream work started."
  exit 2
fi

cd "$REPO" || exit 2

PREP_SAX=$($PY -c '
import json,sys
value=json.load(open(sys.argv[1])).get("source_observations",{}).get("path")
if not isinstance(value,str) or not value: raise SystemExit("native SAX manifest lacks source_observations.path")
print(value)
' "$NATIVE_MANIFEST")
if [ ! -f "$PREP_SAX" ]; then
  echo "Authoritative native-cache source observations are missing: $PREP_SAX"
  exit 1
fi

"$PY" - "$NATIVE_ROOT/sax/manifest.json" "$PREP_SAX" <<'PY'
import hashlib, json, pathlib, sys
manifest, prepared = map(pathlib.Path, sys.argv[1:])
if not manifest.is_file() or not prepared.is_file(): raise SystemExit("native SAX manifest or prepared SAX is missing")
digest = hashlib.sha256(prepared.read_bytes()).hexdigest()
source = json.loads(manifest.read_text()).get("source_observations", {})
if source.get("sha256") != digest: raise SystemExit("REFUSING: L100 native SAX cache does not bind to the reconstructed HHZ prepared SAX")
print("native-cache/prepared-SAX provenance: PASS")
PY
if [ "$?" -ne 0 ]; then exit 1; fi

run_new() {
  label=$1
  output=$2
  shift 2
  if [ -e "$output" ]; then
    if [ -f "$output/manifest.json" ] || [ -f "$output" ]; then
      echo "SKIP existing $label: $output"
      return 0
    fi
    echo "REFUSING partial existing output for $label: $output"
    return 1
  fi
  echo "START $label"
  "$@"
}

mkdir -p "$EVAL_ROOT/bundles/proposed" "$EVAL_ROOT/bundles/fit_first" "$EVAL_ROOT/roi/mapping" "$EVAL_ROOT/metrics" "$EVAL_ROOT/figures"

for CASE in "${CASES[@]}"; do
  CASE_ROOT=$BASE/sparse_inputs/$SUBJECT/$CASE
  MANIFEST=$CASE_ROOT/sparse_manifest.json
  PROPOSED=$BASE/reconstruction/proposed/$SUBJECT/$CASE
  FIT=$BASE/reconstruction/fit_first/$SUBJECT/$CASE
  DIRECT_BUNDLE=$EVAL_ROOT/bundles/proposed/$CASE
  FIT_BUNDLE=$EVAL_ROOT/bundles/fit_first/$CASE
  MAPPING=$EVAL_ROOT/roi/mapping/$CASE.json
  METRICS=$EVAL_ROOT/metrics/$CASE
  FIGURES=$EVAL_ROOT/figures/$CASE

  if [ ! -f "$MANIFEST" ] || [ ! -f "$PROPOSED/model.pt" ] || [ ! -f "$FIT/input_manifest.json" ]; then
    echo "REFUSING $CASE: sparse manifest or completed reconstruction is missing"
    exit 1
  fi

  run_new "$CASE Proposed final-pose bundle" "$DIRECT_BUNDLE" \
    "$PY" -m paper_experiments.joint_vs_2dfit.aha.cyj11_r3_direct_final_pose_bundle \
      --checkpoint "$PROPOSED/model.pt" --prepared-root "$CASE_ROOT" --sparse-manifest "$MANIFEST" \
      --subject "$SUBJECT" --case "$CASE" --output "$DIRECT_BUNDLE" --device cuda:0 || exit 1

  run_new "$CASE Fit-first final-pose bundle" "$FIT_BUNDLE" \
    "$PY" -m paper_experiments.joint_vs_2dfit.aha.current_fit_first_final_pose_bundle \
      --baseline-root "$FIT" --native-maps-root "$NATIVE_ROOT" --subject "$SUBJECT" --case "$CASE" \
      --sparse-manifest "$MANIFEST" --output "$FIT_BUNDLE" || exit 1

  run_new "$CASE AHA mapping" "$MAPPING" \
    "$PY" -m paper_experiments.joint_vs_2dfit.aha.derive_final_pose_mapping \
      --subject "$SUBJECT" --case "$CASE" --prepared-sax "$PREP_SAX" --roi-bundle "$ROI_BUNDLE" --output "$MAPPING" || exit 1

  run_new "$CASE paired AHA and myocardial metrics" "$METRICS" \
    "$PY" -m paper_experiments.joint_vs_2dfit.aha.paired_final_pose_postprocess \
      --subject "$SUBJECT" --case "$CASE" --native-sax-root "$NATIVE_ROOT/sax" --prepared-sax "$PREP_SAX" \
      --mapping "$MAPPING" --roi-bundle "$ROI_BUNDLE" --proposed-bundle "$DIRECT_BUNDLE" --fit-first-bundle "$FIT_BUNDLE" --output "$METRICS" || exit 1

  run_new "$CASE shared-mask through-plane QC" "$FIGURES" \
    "$PY" -m paper_experiments.joint_vs_2dfit.aha.paired_final_pose_through_plane \
      --subject "$SUBJECT" --case "$CASE" --native-sax-root "$NATIVE_ROOT/sax" --prepared-sax "$PREP_SAX" --roi-bundle "$ROI_BUNDLE" \
      --proposed-bundle "$DIRECT_BUNDLE" --fit-first-bundle "$FIT_BUNDLE" \
      --proposed-t1 "$PROPOSED/T1_3D.nii.gz" --proposed-t2 "$PROPOSED/T2_3D.nii.gz" \
      --fit-first-t1 "$FIT/nesvor/t1/volume_ms.nii.gz" --fit-first-t2 "$FIT/nesvor/t2/volume_ms.nii.gz" --output "$FIGURES" || exit 1
done

echo "HHZ paired final-pose downstream evaluation: PASS"
