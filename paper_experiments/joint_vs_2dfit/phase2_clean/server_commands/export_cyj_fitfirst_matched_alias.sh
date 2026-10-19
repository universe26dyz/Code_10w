#!/usr/bin/env bash
# Server-only CYJ Fit-first formal alias. L100 must pass the R0 root for both checkpoint arguments.
set -euo pipefail
[[ $# -eq 9 ]] || { echo "usage: $0 CONDITION ENDPOINT L100_FIT_FIRST_ROOT SPARSE_FIT_FIRST_ROOT SPARSE_MANIFEST R0_NATIVE_MAPS_ROOT NESVOR_EXECUTABLE NEW_OUTPUT DEVICE" >&2; exit 2; }
CONDITION=$1; ENDPOINT=$2; L100=$3; SPARSE=$4; MANIFEST=$5; R0_NATIVE=$6; NESVOR=$7; OUTPUT=$8; DEVICE=$9
L100=$(realpath -m "$L100"); SPARSE=$(realpath -m "$SPARSE"); MANIFEST=$(realpath -m "$MANIFEST"); R0_NATIVE=$(realpath -m "$R0_NATIVE"); NESVOR=$(realpath -m "$NESVOR"); OUTPUT=$(realpath -m "$OUTPUT")
REPO=${REPO:-/data/dengyz/code/Code_10w}
PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
case "$CONDITION" in L100_fold0|L50_fold0|L50_fold1|L25_fold0|L25_fold1|L25_fold2|L25_fold3) ;; *) echo "unsupported CYJ condition: $CONDITION" >&2; exit 2;; esac
case "$ENDPOINT" in central|psf128) ;; *) echo "unsupported endpoint: $ENDPOINT" >&2; exit 2;; esac
test -d "$L100" && test -d "$SPARSE" && test -f "$MANIFEST" && test -d "$R0_NATIVE" && test -x "$NESVOR"
test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
printf 'CYJ Fit-first formal alias:\n  condition=%s\n  endpoint=%s\n  l100_root=%s\n  sparse_root=%s\n  manifest=%s\n  r0_native=%s\n  output=%s\n  device=%s\n' "$CONDITION" "$ENDPOINT" "$L100" "$SPARSE" "$MANIFEST" "$R0_NATIVE" "$OUTPUT" "$DEVICE"
cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj_matched_comparison export-fit-first --condition "$CONDITION" --endpoint "$ENDPOINT" --l100-fit-first-root "$L100" --sparse-fit-first-root "$SPARSE" --sparse-manifest "$MANIFEST" --r0-native-maps-root "$R0_NATIVE" --nesvor-executable "$NESVOR" --output "$OUTPUT" --device "$DEVICE"
