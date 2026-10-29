#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: bash $0 PROPOSED_RUN FIT_FIRST_RUN ROI_BUNDLE SUBJECT [CASE ...]" >&2
  exit 2
fi

PROPOSED_RUN=$1
FIT_FIRST_RUN=$2
ROI_BUNDLE=$3
SUBJECT=$4
shift 4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SUBJECT="$SUBJECT" bash "$SCRIPT_DIR/hhz_paired_final_pose.sh" \
  "$PROPOSED_RUN" "$FIT_FIRST_RUN" "$ROI_BUNDLE" "$@"
