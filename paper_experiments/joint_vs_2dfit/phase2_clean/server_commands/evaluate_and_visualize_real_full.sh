#!/usr/bin/env bash
set -euo pipefail
# Server-only, read-only reconstruction consumer: write a new R0 evaluation plus figures.
[[ $# -eq 4 ]] || { echo "usage: $0 PROPOSED_ROOT FIT_FIRST_ROOT NEW_EVALUATION_ROOT NEW_FIGURE_ROOT" >&2; exit 2; }
PROPOSED=$1; FIT=$2; EVALUATION=$3; FIGURES=$4
DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}; REPO=${REPO:-/data/dengyz/code/Code_10w}; PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
test ! -e "$EVALUATION" || { echo "refusing overwrite: $EVALUATION" >&2; exit 1; }
test ! -e "$FIGURES" || { echo "refusing overwrite: $FIGURES" >&2; exit 1; }
cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.visualization --mode r0-preflight --proposed-root "$PROPOSED" --fit-first-root "$FIT" --data-root "$DATA" >/dev/null
EVALUATION_PARENT=$(dirname "$EVALUATION"); EVALUATION_NAME=$(basename "$EVALUATION")
FIGURES_PARENT=$(dirname "$FIGURES"); FIGURES_NAME=$(basename "$FIGURES")
mkdir -p "$EVALUATION_PARENT" "$FIGURES_PARENT"
EVALUATION_STAGING=$(mktemp -d "$EVALUATION_PARENT/.${EVALUATION_NAME}.staging-XXXXXX")
# evaluate_real_full.sh owns its own atomic staging directory and therefore
# requires its destination to be absent when invoked.
rmdir "$EVALUATION_STAGING"
FIGURES_STAGING=$(mktemp -d "$FIGURES_PARENT/.${FIGURES_NAME}.staging-XXXXXX")
cleanup() { status=$?; if [[ -d "$EVALUATION_STAGING" ]]; then rm -rf -- "$EVALUATION_STAGING"; fi; if [[ -d "$FIGURES_STAGING" ]]; then rm -rf -- "$FIGURES_STAGING"; fi; exit "$status"; }
trap cleanup EXIT
"$REPO/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/evaluate_real_full.sh" "$PROPOSED" "$FIT" "$EVALUATION_STAGING"
for SID in CYJ DYZ HHZ HJL; do
  REF="$DATA/reference_data/phase2_clean/$SID/native_reference_2d"
  MASK="$DATA/reference_data/phase2_clean/$SID/myocardium_masks_v1"
  "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.visualization --mode r0 --subject "$SID" --native-reference "$REF" --myocardium-bundle "$MASK" --proposed-t1 "$PROPOSED/$SID/T1_3D.nii.gz" --proposed-t2 "$PROPOSED/$SID/T2_3D.nii.gz" --fit-first-t1 "$FIT/$SID/nesvor/t1/volume_ms.nii.gz" --fit-first-t2 "$FIT/$SID/nesvor/t2/volume_ms.nii.gz" --output "$FIGURES_STAGING/$SID"
done
mv "$EVALUATION_STAGING" "$EVALUATION"
EVALUATION_STAGING=
mv "$FIGURES_STAGING" "$FIGURES"
FIGURES_STAGING=
