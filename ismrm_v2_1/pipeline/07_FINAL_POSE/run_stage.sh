#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT=${1:?usage: $0 RUN_ROOT SUBJECT ROI_BUNDLE CASE}
SUBJECT=${2:?usage: $0 RUN_ROOT SUBJECT ROI_BUNDLE CASE}
ROI_BUNDLE=${3:?usage: $0 RUN_ROOT SUBJECT ROI_BUNDLE CASE}
CASE=${4:?usage: $0 RUN_ROOT SUBJECT ROI_BUNDLE CASE}
REPO="$PROJECT_ROOT/common" SUBJECT="$SUBJECT" \
  bash "$REPO/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_paired_final_pose.sh" \
  "$RUN_ROOT" "$RUN_ROOT" "$ROI_BUNDLE" "$CASE"

