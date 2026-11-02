#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT=${1:?usage: $0 RUN_ROOT SUBJECT MANIFEST [MATLAB] [PYTHON]}
SUBJECT=${2:?usage: $0 RUN_ROOT SUBJECT MANIFEST [MATLAB] [PYTHON]}
MANIFEST=${3:?usage: $0 RUN_ROOT SUBJECT MANIFEST [MATLAB] [PYTHON]}
MATLAB=${4:-matlab}; PYTHON=${5:-python}
REPO="$PROJECT_ROOT/common" \
  bash "$REPO/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/server_preprocess_and_prepare.sh" \
  "$RUN_ROOT" "$SUBJECT" "$MANIFEST" "$PROJECT_ROOT/configs/protocol_scmr_v5_10_100.yaml" "$MATLAB" "$PYTHON"

