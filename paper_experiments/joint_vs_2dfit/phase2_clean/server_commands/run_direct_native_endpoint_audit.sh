#!/usr/bin/env bash
# Server-only, read-only canonical Q004 central endpoint audit.
set -euo pipefail

[[ $# -eq 11 ]] || { echo "usage: $0 CHECKPOINT CROPPED_PREPARED_ROOT R0_NATIVE_MAPS_ROOT L100_SPARSE_MANIFEST CENTRAL_T1 CENTRAL_T2 L100_MASK_T1 L100_MASK_T2 NEW_OUTPUT_ROOT DEVICE PYTHON" >&2; exit 2; }
CHECKPOINT=$(realpath -m "$1"); PREPARED=$(realpath -m "$2"); R0_NATIVE=$(realpath -m "$3"); SPARSE=$(realpath -m "$4")
CENTRAL_T1=$(realpath -m "$5"); CENTRAL_T2=$(realpath -m "$6"); MASK_T1=$(realpath -m "$7"); MASK_T2=$(realpath -m "$8")
OUTPUT=$(realpath -m "$9"); DEVICE=${10}; PY=${11}
for path in "$CHECKPOINT" "$SPARSE" "$CENTRAL_T1" "$CENTRAL_T2" "$MASK_T1" "$MASK_T2"; do test -f "$path"; done
test -d "$PREPARED"; test -d "$R0_NATIVE"; test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
REPO=${REPO:-/data/dengyz/code/Code_10w}; cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.direct_native_endpoint_audit --checkpoint "$CHECKPOINT" --prepared-root "$PREPARED" --r0-native-maps-root "$R0_NATIVE" --sparse-manifest "$SPARSE" --central-t1 "$CENTRAL_T1" --central-t2 "$CENTRAL_T2" --l100-t1 "$MASK_T1" --l100-t2 "$MASK_T2" --output "$OUTPUT" --device "$DEVICE"
