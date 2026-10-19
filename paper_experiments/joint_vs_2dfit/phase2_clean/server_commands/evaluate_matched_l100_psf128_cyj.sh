#!/usr/bin/env bash
# Server-only PSF128 matched endpoint evaluation. Do not run from local Codex.
set -euo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 SPARSE_MANIFEST R0_NATIVE_MAPS_ROOT PROPOSED_EXPORT_ROOT FIT_FIRST_EXPORT_ROOT NEW_OUTPUT" >&2; exit 2; }
MANIFEST=$1; R0_NATIVE=$2; PROPOSED=$3; FIT_FIRST=$4; OUTPUT=$5
MANIFEST=$(realpath -m "$MANIFEST"); R0_NATIVE=$(realpath -m "$R0_NATIVE"); PROPOSED=$(realpath -m "$PROPOSED"); FIT_FIRST=$(realpath -m "$FIT_FIRST"); OUTPUT=$(realpath -m "$OUTPUT")
REPO=${REPO:-/data/dengyz/code/Code_10w}
PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
SUBJECT=${SUBJECT:-CYJ}; CONDITION=${CONDITION:-L25_fold0}; R0_RUN_ID=${R0_RUN_ID:-R0_PHASE4R}; R1_RUN_ID=${R1_RUN_ID:-R1_PHASE4R}
test -f "$MANIFEST" && test -d "$R0_NATIVE" && test -d "$PROPOSED" && test -d "$FIT_FIRST"
test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
printf 'PSF128 evaluation inputs:\n  manifest=%s\n  r0_native=%s\n  proposed=%s\n  fit_first=%s\n  output=%s\n' "$MANIFEST" "$R0_NATIVE" "$PROPOSED" "$FIT_FIRST" "$OUTPUT"
cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.l100_mask_paired_eval --subject "$SUBJECT" --condition "$CONDITION" --r0-run-id "$R0_RUN_ID" --r1-run-id "$R1_RUN_ID" --sparse-manifest "$MANIFEST" --r0-native-maps-root "$R0_NATIVE" --endpoint psf128 --proposed-export-root "$PROPOSED" --fit-first-export-root "$FIT_FIRST" --output "$OUTPUT"
