#!/usr/bin/env bash
set -euo pipefail
# Server-only R0 agreement evaluation and subject-level aggregation.
[[ $# -eq 3 ]] || { echo "usage: $0 PROPOSED_ROOT FIT_FIRST_ROOT NEW_OUTPUT_ROOT" >&2; exit 2; }
PROPOSED=$1; FIT=$2; OUTPUT=$3; DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}; REPO=${REPO:-/data/dengyz/code/Code_10w}; PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.visualization --mode r0-preflight --proposed-root "$PROPOSED" --fit-first-root "$FIT" --data-root "$DATA" >/dev/null
OUTPUT_PARENT=$(dirname "$OUTPUT"); OUTPUT_NAME=$(basename "$OUTPUT")
mkdir -p "$OUTPUT_PARENT"
STAGING=$(mktemp -d "$OUTPUT_PARENT/.${OUTPUT_NAME}.staging-XXXXXX")
cleanup() { status=$?; if [[ -d "$STAGING" ]]; then rm -rf -- "$STAGING"; fi; exit "$status"; }
trap cleanup EXIT
for SID in CYJ DYZ HHZ HJL; do
  REF="$DATA/reference_data/phase2_clean/$SID/native_reference_2d"; MASK="$DATA/reference_data/phase2_clean/$SID/myocardium_masks_v1"
  "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.evaluation --mode real-paired --subject "$SID" --proposed-t1 "$PROPOSED/$SID/T1_3D.nii.gz" --proposed-t2 "$PROPOSED/$SID/T2_3D.nii.gz" --fit-first-t1 "$FIT/$SID/nesvor/t1/volume_ms.nii.gz" --fit-first-t2 "$FIT/$SID/nesvor/t2/volume_ms.nii.gz" --native-reference "$REF" --myocardium-bundle "$MASK" --output "$STAGING/${SID}_paired.json"
done
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.aggregate_real --r0-evaluation-root "$STAGING" --output-root "$STAGING/cohort"
mv "$STAGING" "$OUTPUT"
STAGING=
