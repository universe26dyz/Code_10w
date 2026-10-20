#!/usr/bin/env bash
# Server-only read-only conversion of explicit per-plane legacy inputs to masks.
set -euo pipefail
[[ $# -eq 3 ]] || { echo "usage: $0 INPUT_BUNDLE_ROOT NEW_MASK_BUNDLE_ROOT PYTHON" >&2; exit 2; }
INPUT=$(realpath -m "$1"); OUTPUT=$(realpath -m "$2"); PY=$3
test -d "$INPUT"; test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
REPO=${REPO:-/data/dengyz/code/Code_10w}; cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.scmr_central_component_mask --input-root "$INPUT" --output "$OUTPUT"
