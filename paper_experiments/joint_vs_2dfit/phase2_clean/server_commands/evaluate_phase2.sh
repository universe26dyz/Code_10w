#!/usr/bin/env bash
set -euo pipefail
# not auto-launched: formal layout is explicit: proposed roots contain T1_3D.nii.gz/T2_3D.nii.gz;
# fit-first roots contain nesvor/{t1,t2}/volume_ms.nii.gz.  Real roots contain
# one such method root per CYJ,DYZ,HHZ,HJL subject.
[[ $# -eq 5 ]] || { echo "usage: $0 PROPOSED_SYN_ROOT FIT_SYN_ROOT PROPOSED_REAL_ROOT FIT_REAL_ROOT NEW_OUTPUT_ROOT" >&2; exit 2; }
"$(dirname "$0")/server_preflight_phase2.sh" >/tmp/phase2_preflight.json
REPO=${REPO:-/data/dengyz/code/Code_10w}; DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}; cd "$REPO"
PROPOSED_SYN=$1; FIT_SYN=$2; PROPOSED_REAL=$3; FIT_REAL=$4; OUTPUT=$5
[[ ! -e "$OUTPUT" ]] || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
mkdir -p "$OUTPUT"
GT=paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_1_clean_mrxcat/gt_maps.npz
MESH=paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_1_clean_mrxcat/mesh_primary_myo.npz
python -m paper_experiments.joint_vs_2dfit.phase2_clean.evaluation --mode synthetic --synthetic-t1 "$PROPOSED_SYN/T1_3D.nii.gz" --synthetic-t2 "$PROPOSED_SYN/T2_3D.nii.gz" --gt-maps "$GT" --mesh-myo "$MESH" --output "$OUTPUT/synthetic_proposed.json"
python -m paper_experiments.joint_vs_2dfit.phase2_clean.evaluation --mode synthetic --synthetic-t1 "$FIT_SYN/nesvor/t1/volume_ms.nii.gz" --synthetic-t2 "$FIT_SYN/nesvor/t2/volume_ms.nii.gz" --gt-maps "$GT" --mesh-myo "$MESH" --output "$OUTPUT/synthetic_fit_first.json"
for SID in CYJ DYZ HHZ HJL; do
  REF="$DATA/reference_data/phase2_clean/$SID/native_reference_2d"
  MASK="$DATA/reference_data/phase2_clean/$SID/myocardium_masks_v1"
  python -m paper_experiments.joint_vs_2dfit.phase2_clean.evaluation --mode real --subject "$SID" --real-t1 "$PROPOSED_REAL/$SID/T1_3D.nii.gz" --real-t2 "$PROPOSED_REAL/$SID/T2_3D.nii.gz" --native-reference "$REF" --myocardium-bundle "$MASK" --output "$OUTPUT/${SID}_proposed.json"
  python -m paper_experiments.joint_vs_2dfit.phase2_clean.evaluation --mode real --subject "$SID" --real-t1 "$FIT_REAL/$SID/nesvor/t1/volume_ms.nii.gz" --real-t2 "$FIT_REAL/$SID/nesvor/t2/volume_ms.nii.gz" --native-reference "$REF" --myocardium-bundle "$MASK" --output "$OUTPUT/${SID}_fit_first.json"
done
