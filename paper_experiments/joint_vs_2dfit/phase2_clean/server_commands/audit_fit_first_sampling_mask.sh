#!/usr/bin/env bash
# Read-only sampling-mask audit; this never invokes NeSVoR reconstruction or training.
set -euo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 R0_FIT_FIRST_SUBJECT_ROOT SPARSE_FIT_FIRST_ROOT SPARSE_MANIFEST R0_NATIVE_MAPS_ROOT NEW_OUTPUT" >&2; exit 2; }
L100=$1
SPARSE=$2
MANIFEST=$3
R0_NATIVE=$4
OUTPUT=$5
REPO=${REPO:-/data/dengyz/code/Code_10w}
PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
NESVOR=${NESVOR_EXECUTABLE:-/home/dengyz/miniconda3/envs/nesvor/bin/nesvor}
DEVICE=${PHASE4R_AUDIT_DEVICE:-0}
test -d "$L100" && test -d "$SPARSE" && test -f "$MANIFEST" && test -d "$R0_NATIVE" && test -x "$NESVOR"
test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_mask_audit \
  --l100-fit-first-root "$L100" --sparse-fit-first-root "$SPARSE" \
  --sparse-manifest "$MANIFEST" --r0-native-maps-root "$R0_NATIVE" \
  --nesvor-executable "$NESVOR" --output "$OUTPUT" --device "$DEVICE"
