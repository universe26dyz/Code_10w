#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT=${1:?usage: $0 RUN_ROOT SUBJECT [MATLAB]}
SUBJECT=${2:?usage: $0 RUN_ROOT SUBJECT [MATLAB]}
MATLAB=${3:-matlab}; PYTHON=${PYTHON:-python}; NESVOR=${NESVOR:-nesvor}
REPO="$PROJECT_ROOT/common" PY="$PYTHON" NESVOR="$NESVOR" \
  bash "$REPO/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh" \
  "$RUN_ROOT" "$RUN_ROOT/preprocessed" "$SUBJECT" "$RUN_ROOT/prepared" "$MATLAB"

