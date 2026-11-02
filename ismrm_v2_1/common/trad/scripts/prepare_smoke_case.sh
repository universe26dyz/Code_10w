#!/usr/bin/env bash
set -euo pipefail

MAT_PATH="/tmp/multimap_phase2_smoke.xI94AL/CYJ_multimap_2ch_901_v1.mat"
DICOM_DIR="/home/universe/SVR/multimap_postprogramming/origin_data/CYJ_20260819_163756/multimap_2ch_901"
OUTPUT_DIR="${1:?usage: prepare_smoke_case.sh /absolute/output_dir}"

if [[ ! -f "${MAT_PATH}" || ! -d "${DICOM_DIR}" ]]; then
  echo "Required Phase-2 CYJ 2ch MAT or DICOM directory is unavailable; do not regenerate preprocessing." >&2
  exit 1
fi

conda run -n knesvr_torch python -m modules.module_02_data_bridge.prepare_observations \
  --preprocessed-mat "${MAT_PATH}" \
  --dicom-dir "${DICOM_DIR}" \
  --stack 2ch \
  --output-dir "${OUTPUT_DIR}" \
  --max-groups 1
