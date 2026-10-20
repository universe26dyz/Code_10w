#!/usr/bin/env bash
# Server-only immutable direct-native bundle launcher. Run inside tmux; safe to rerun after publication.
set -euo pipefail

[[ $# -eq 7 ]] || { echo "usage: $0 CHECKPOINT CROPPED_PREPARED_ROOT ENDPOINT NEW_OUTPUT_ROOT DEVICE PYTHON QUERY_BATCH_SIZE" >&2; exit 2; }
CHECKPOINT=$(realpath -m "$1"); PREPARED=$(realpath -m "$2"); ENDPOINT=$3; OUTPUT=$(realpath -m "$4")
DEVICE=$5; PY=$6; QUERY_BATCH_SIZE=$7
[[ "$ENDPOINT" == central || "$ENDPOINT" == psf128 ]] || { echo "ENDPOINT must be central or psf128" >&2; exit 2; }
test -f "$CHECKPOINT"; test -d "$PREPARED"
if test -e "$OUTPUT"; then
  test -f "$OUTPUT/manifest.json" && { echo "already published: $OUTPUT"; exit 0; }
  echo "refusing non-published existing output: $OUTPUT" >&2; exit 1
fi
REPO=${REPO:-/data/dengyz/code/Code_10w}; LOG=${OUTPUT}.launch.log
mkdir -p "$(dirname "$OUTPUT")"; cd "$REPO"
printf 'direct-native bundle: endpoint=%s checkpoint=%s prepared=%s output=%s device=%s batch=%s\n' "$ENDPOINT" "$CHECKPOINT" "$PREPARED" "$OUTPUT" "$DEVICE" "$QUERY_BATCH_SIZE" | tee -a "$LOG"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.direct_native_bundles --checkpoint "$CHECKPOINT" --prepared-root "$PREPARED" --endpoint "$ENDPOINT" --output "$OUTPUT" --device "$DEVICE" --query-batch-size "$QUERY_BATCH_SIZE" 2>&1 | tee -a "$LOG"
