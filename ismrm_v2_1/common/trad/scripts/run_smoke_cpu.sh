#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/universe/SVR/multimap_postprogramming/Code_10w/trad"
SMOKE_ROOT="/tmp/multimap_phase4_trad_smoke"
PREPARED_DIR="${SMOKE_ROOT}/prepared_case"
OUTPUT_DIR="${SMOKE_ROOT}/outputs"

if [[ -e "${SMOKE_ROOT}" ]]; then
  echo "Smoke root already exists; refusing to overwrite: ${SMOKE_ROOT}" >&2
  exit 1
fi
mkdir -p "${SMOKE_ROOT}"
bash "${PROJECT_ROOT}/scripts/prepare_smoke_case.sh" "${PREPARED_DIR}"
cd "${PROJECT_ROOT}"
conda run -n knesvr_torch python -m scripts.run_training \
  --config configs/smoke_cpu.yaml \
  --protocol configs/protocol_hhz_v1.yaml \
  --observations "${PREPARED_DIR}/observations.npz" \
  --output-dir "${OUTPUT_DIR}"
