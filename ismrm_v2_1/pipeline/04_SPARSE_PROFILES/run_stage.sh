#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT=${1:?usage: $0 RUN_ROOT SUBJECT [PYTHON]}
SUBJECT=${2:?usage: $0 RUN_ROOT SUBJECT [PYTHON]}
PYTHON=${3:-python}
REPO="$PROJECT_ROOT/common" \
  bash "$REPO/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/build_sparse_profiles.sh" \
  "$RUN_ROOT/prepared" "$SUBJECT" "$RUN_ROOT/sparse_inputs" "$PYTHON"

