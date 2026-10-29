#!/usr/bin/env bash
set -euo pipefail

# Generic subject entry point. The retained historical implementation remains
# the execution body; SUBJECT is explicit here and never inferred from a path.
if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
  echo "Usage: bash $0 RUN_ROOT PREPROCESSED_ROOT SUBJECT [PREPARED_ROOT] [MATLAB]" >&2
  exit 2
fi

RUN_ROOT=$1
PREPROCESSED_ROOT=$2
SUBJECT=$3
shift 3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SUBJECT="$SUBJECT" bash "$SCRIPT_DIR/run_hhz_scmr_v5_fit_first.sh" \
  "$RUN_ROOT" "$PREPROCESSED_ROOT" "$@"
