#!/usr/bin/env bash
# Server-only immutable final-pose Fit-first endpoint. Requires a passing v2 audit.
set -euo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 AUDIT_ROOT BASELINE_ROOT R0_NATIVE_MAPS_ROOT NEW_BUNDLE_ROOT PYTHON" >&2; exit 2; }
AUDIT=$(realpath -m "$1"); BASELINE=$(realpath -m "$2"); R0=$(realpath -m "$3"); OUTPUT=$(realpath -m "$4"); PY=$5
test -f "$AUDIT/manifest.json"; test -d "$BASELINE"; test -d "$R0"
if test -e "$OUTPUT"; then test -f "$OUTPUT/manifest.json" && { echo "already published: $OUTPUT"; exit 0; }; echo "refusing existing output: $OUTPUT" >&2; exit 1; fi
REPO=${REPO:-/data/dengyz/code/Code_10w}; mkdir -p "$(dirname "$OUTPUT")"; cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_native_bundles --audit-root "$AUDIT" --baseline-root "$BASELINE" --r0-native-maps-root "$R0" --output "$OUTPUT"
