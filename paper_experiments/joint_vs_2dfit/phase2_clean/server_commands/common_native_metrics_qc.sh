#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "Usage: bash $0 PROPOSED_RUN FIT_FIRST_RUN ROI_BUNDLE SUBJECT CASE [CASE ...]" >&2
  exit 2
fi

PROPOSED_RUN=$1
FIT_FIRST_RUN=$2
ROI_BUNDLE=$3
SUBJECT=$4
shift 4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SUBJECT="$SUBJECT" bash "$SCRIPT_DIR/hhz_common_native_metrics_qc.sh" \
  "$PROPOSED_RUN" "$FIT_FIRST_RUN" "$ROI_BUNDLE" "$@"
