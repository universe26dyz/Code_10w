#!/usr/bin/env bash
# Server-only historical-SCMR producer. It only reads completed artifacts.
set -euo pipefail
[[ $# -ge 7 ]] || { echo "usage: $0 discover|build HISTORICAL_COMMON_MASK_MANIFEST GROUP_MAP_MANIFEST R0_NATIVE_ROOT L100_SPARSE_MANIFEST PHASE4R_L100_NATIVE_SUPPORT_ROOT PYTHON [NEW_OUTPUT_ROOT]" >&2; exit 2; }
MODE=$1; COMMON=$(realpath -m "$2"); GROUPS=$(realpath -m "$3"); R0=$(realpath -m "$4"); SPARSE=$(realpath -m "$5"); L100=$(realpath -m "$6"); PY=$7; OUTPUT=${8:-}
for path in "$COMMON" "$GROUPS" "$SPARSE" "$L100/manifest.json"; do test -f "$path" || { echo "missing prerequisite: $path" >&2; exit 1; }; done
test -d "$R0" || { echo "missing prerequisite: $R0" >&2; exit 1; }
REPO=${REPO:-/data/dengyz/code/Code_10w}; cd "$REPO"
ARGS=(--historical-common-mask-manifest "$COMMON" --group-map-manifest "$GROUPS" --r0-native-maps-root "$R0" --sparse-manifest "$SPARSE" --phase4r-l100-root "$L100")
case "$MODE" in
  discover) [[ -z "$OUTPUT" ]] || { echo "discover takes no output root" >&2; exit 2; }; "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.scmr_six_input_bundle --mode discover "${ARGS[@]}" ;;
  build) [[ -n "$OUTPUT" ]] || { echo "build requires NEW_OUTPUT_ROOT" >&2; exit 2; }; OUTPUT=$(realpath -m "$OUTPUT"); test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }; mkdir -p "$(dirname "$OUTPUT")"; "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.scmr_six_input_bundle --mode build "${ARGS[@]}" --output "$OUTPUT" ;;
  *) echo "mode must be discover or build" >&2; exit 2 ;;
esac
