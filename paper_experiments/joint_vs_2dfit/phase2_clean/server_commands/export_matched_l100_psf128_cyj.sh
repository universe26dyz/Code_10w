#!/usr/bin/env bash
# Server-only immutable post-hoc PSF128 matched-output export. Do not run from local Codex.
set -euo pipefail
[[ $# -eq 7 ]] || { echo "usage: $0 L100_FIT_FIRST_ROOT SPARSE_FIT_FIRST_ROOT SPARSE_MANIFEST R0_NATIVE_MAPS_ROOT PROPOSED_SOURCE_ROOT FIT_FIRST_NEW_OUTPUT PROPOSED_NEW_OUTPUT" >&2; exit 2; }
L100=$1; SPARSE=$2; MANIFEST=$3; R0_NATIVE=$4; PROPOSED=$5; FIT_OUTPUT=$6; PROPOSED_OUTPUT=$7
L100=$(realpath -m "$L100"); SPARSE=$(realpath -m "$SPARSE"); MANIFEST=$(realpath -m "$MANIFEST"); R0_NATIVE=$(realpath -m "$R0_NATIVE"); PROPOSED=$(realpath -m "$PROPOSED"); FIT_OUTPUT=$(realpath -m "$FIT_OUTPUT"); PROPOSED_OUTPUT=$(realpath -m "$PROPOSED_OUTPUT")
REPO=${REPO:-/data/dengyz/code/Code_10w}
PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
NESVOR=${NESVOR_EXECUTABLE:-/home/dengyz/miniconda3/envs/nesvor/bin/nesvor}
DEVICE=${PHASE4R_EXPORT_DEVICE:-0}
SUBJECT=${SUBJECT:-CYJ}; CONDITION=${CONDITION:-L100_fold0}
test -d "$L100" && test -d "$SPARSE" && test -f "$MANIFEST" && test -d "$R0_NATIVE" && test -d "$PROPOSED" && test -x "$NESVOR"
test ! -e "$FIT_OUTPUT" || { echo "refusing overwrite: $FIT_OUTPUT" >&2; exit 1; }
test ! -e "$PROPOSED_OUTPUT" || { echo "refusing overwrite: $PROPOSED_OUTPUT" >&2; exit 1; }
printf 'PSF128 endpoint inputs:\n  L100=%s\n  sparse=%s\n  manifest=%s\n  r0_native=%s\n  proposed=%s\n  fit_output=%s\n  proposed_output=%s\n' "$L100" "$SPARSE" "$MANIFEST" "$R0_NATIVE" "$PROPOSED" "$FIT_OUTPUT" "$PROPOSED_OUTPUT"
cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.matched_output proposed --subject "$SUBJECT" --condition "$CONDITION" --source-root "$PROPOSED" --endpoint psf128 --output "$PROPOSED_OUTPUT"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.matched_output fit-first --subject "$SUBJECT" --condition "$CONDITION" --l100-fit-first-root "$L100" --sparse-fit-first-root "$SPARSE" --sparse-manifest "$MANIFEST" --r0-native-maps-root "$R0_NATIVE" --nesvor-executable "$NESVOR" --endpoint psf128 --output "$FIT_OUTPUT" --device "$DEVICE"
