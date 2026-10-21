#!/usr/bin/env bash
# Server-only immutable final-pose Fit-first endpoint. Requires a passing v3 audit.
set -euo pipefail
[[ $# -eq 7 ]] || { echo "usage: $0 AUDIT_ROOT BASELINE_ROOT R0_NATIVE_MAPS_ROOT ENDPOINT ENDPOINT_SOURCE_ROOT NEW_BUNDLE_ROOT PYTHON" >&2; exit 2; }
AUDIT=$(realpath -m "$1"); BASELINE=$(realpath -m "$2"); R0=$(realpath -m "$3"); ENDPOINT=$4; SOURCE=$(realpath -m "$5"); OUTPUT=$(realpath -m "$6"); PY=$7
case "$ENDPOINT" in central|psf128) ;; *) echo "endpoint must be central or psf128" >&2; exit 2;; esac
test -f "$AUDIT/manifest.json"; test -d "$BASELINE"; test -d "$R0"; test -f "$SOURCE/manifest.json"
if test -e "$OUTPUT"; then test -f "$OUTPUT/manifest.json" && { echo "already published: $OUTPUT"; exit 0; }; echo "refusing existing output: $OUTPUT" >&2; exit 1; fi
REPO=${REPO:-/data/dengyz/code/Code_10w}; mkdir -p "$(dirname "$OUTPUT")"; cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_native_bundles --audit-root "$AUDIT" --baseline-root "$BASELINE" --r0-native-maps-root "$R0" --endpoint "$ENDPOINT" --endpoint-source-root "$SOURCE" --output "$OUTPUT"
