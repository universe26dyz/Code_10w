#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SUBJECT=${SUBJECT:?set SUBJECT, e.g. HHZ}
DRY_RUN=${DRY_RUN:-1}

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "SUBJECT=$SUBJECT"
echo "EXECUTION_HOST=$(hostname)"
echo "PROTOCOL=$PROJECT_ROOT/configs/protocol_scmr_v5_10_100.yaml"
echo "DECODER=$PROJECT_ROOT/runtime/model_assets/decoder_scmr_v5_10_100.pth"
echo "B6_MODEL=$PROJECT_ROOT/runtime/model_assets/b6_model.pt"
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY-RUN: no MATLAB, CUDA or reconstruction command launched"
  exit 0
fi
echo "Use the nine module manuals in order; full execution requires subject data and runtime assets."

