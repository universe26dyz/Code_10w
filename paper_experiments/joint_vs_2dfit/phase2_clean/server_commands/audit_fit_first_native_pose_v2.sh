#!/usr/bin/env bash
# Server-only, read-only Fit-first v3 geometry gate. Never launches reconstruction.
set -euo pipefail
[[ $# -eq 4 ]] || { echo "usage: $0 BASELINE_ROOT R0_NATIVE_MAPS_ROOT NEW_AUDIT_ROOT PYTHON" >&2; exit 2; }
BASELINE=$(realpath -m "$1"); R0=$(realpath -m "$2"); OUTPUT=$(realpath -m "$3"); PY=$4
test -d "$BASELINE"; test -d "$R0"
if test -e "$OUTPUT"; then test -f "$OUTPUT/manifest.json" && { echo "already published (choose a new v3 artifact root): $OUTPUT"; exit 0; }; echo "refusing existing output: $OUTPUT" >&2; exit 1; fi
REPO=${REPO:-/data/dengyz/code/Code_10w}; mkdir -p "$(dirname "$OUTPUT")"; cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_native_pose_audit --baseline-root "$BASELINE" --r0-native-maps-root "$R0" --output "$OUTPUT"
