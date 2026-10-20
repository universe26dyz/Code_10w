#!/usr/bin/env bash
# Server-only immutable common-support metrics for direct and Fit-first bundles.
set -euo pipefail
[[ $# -eq 7 ]] || { echo "usage: $0 DIRECT_BUNDLE FIT_FIRST_BUNDLE R0_NATIVE SPARSE_MANIFEST MASK_ROOT NEW_OUTPUT_ROOT PYTHON" >&2; exit 2; }
DIRECT=$(realpath -m "$1"); FIT=$(realpath -m "$2"); R0=$(realpath -m "$3"); SPARSE=$(realpath -m "$4"); MASKS=$(realpath -m "$5"); OUTPUT=$(realpath -m "$6"); PY=$7
test -f "$DIRECT/manifest.json"; test -f "$FIT/manifest.json"; test -d "$R0"; test -f "$SPARSE"; test -d "$MASKS"
if test -e "$OUTPUT"; then test -f "$OUTPUT/manifest.json" && { echo "already published: $OUTPUT"; exit 0; }; echo "refusing existing output: $OUTPUT" >&2; exit 1; fi
REPO=${REPO:-/data/dengyz/code/Code_10w}; mkdir -p "$(dirname "$OUTPUT")"; cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.final_pose_native_metrics --direct-bundle-root "$DIRECT" --fit-first-bundle-root "$FIT" --r0-native-maps-root "$R0" --sparse-manifest "$SPARSE" --mask-root "$MASKS" --output "$OUTPUT"
