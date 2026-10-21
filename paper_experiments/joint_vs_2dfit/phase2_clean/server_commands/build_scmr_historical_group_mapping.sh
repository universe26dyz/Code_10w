#!/usr/bin/env bash
# Server-only, read-only identity audit for the historical SCMR six-input source.
set -euo pipefail
[[ $# -ge 5 ]] || { echo "usage: $0 discover|build HISTORICAL_COMMON_MASK_MANIFEST R0_NATIVE_ROOT L100_SPARSE_MANIFEST PYTHON [NEW_OUTPUT_ROOT]" >&2; exit 2; }
MODE=$1; COMMON=$(realpath -m "$2"); R0=$(realpath -m "$3"); SPARSE=$(realpath -m "$4"); PY=$5; OUTPUT=${6:-}
for path in "$COMMON" "$SPARSE"; do test -f "$path" || { echo "missing prerequisite: $path" >&2; exit 1; }; done
test -d "$R0" || { echo "missing prerequisite: $R0" >&2; exit 1; }
REPO=${REPO:-/data/dengyz/code/Code_10w}; cd "$REPO"
ARGS=(--historical-common-mask-manifest "$COMMON" --r0-native-maps-root "$R0" --sparse-manifest "$SPARSE")
case "$MODE" in
  discover) [[ -z "$OUTPUT" ]] || { echo "discover takes no output root" >&2; exit 2; }; "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.scmr_historical_group_mapping --mode discover "${ARGS[@]}" ;;
  build) [[ -n "$OUTPUT" ]] || { echo "build requires NEW_OUTPUT_ROOT" >&2; exit 2; }; OUTPUT=$(realpath -m "$OUTPUT"); test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }; mkdir -p "$(dirname "$OUTPUT")"; "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.scmr_historical_group_mapping --mode build "${ARGS[@]}" --output "$OUTPUT" ;;
  *) echo "mode must be discover or build" >&2; exit 2 ;;
esac
