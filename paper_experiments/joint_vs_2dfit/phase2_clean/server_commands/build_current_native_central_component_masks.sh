#!/usr/bin/env bash
# Server-only current-native formal-mask producer; no training or reconstruction.
set -euo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 R0_NATIVE_MAPS_ROOT L100_SPARSE_MANIFEST FROZEN_L100_NATIVE_SUPPORT_ROOT NEW_OUTPUT_ROOT PYTHON" >&2; exit 2; }
R0=$(realpath -m "$1"); SPARSE=$(realpath -m "$2"); L100=$(realpath -m "$3"); OUTPUT=$(realpath -m "$4"); PY=$5
test -d "$R0" || { echo "missing prerequisite: $R0" >&2; exit 1; }
for path in "$SPARSE" "$L100/manifest.json"; do test -f "$path" || { echo "missing prerequisite: $path" >&2; exit 1; }; done
test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
REPO=${REPO:-/data/dengyz/code/Code_10w}; mkdir -p "$(dirname "$OUTPUT")"; cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.current_native_central_component_mask --r0-native-maps-root "$R0" --sparse-manifest "$SPARSE" --l100-native-support-root "$L100" --output "$OUTPUT"
