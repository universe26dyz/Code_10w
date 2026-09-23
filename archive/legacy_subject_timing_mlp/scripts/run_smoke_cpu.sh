#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/universe/SVR/multimap_postprogramming/Code_10w/mlp"
PREPARED_NPZ="/tmp/multimap_phase4_trad_smoke/prepared_case/observations.npz"
SMOKE_ROOT="/tmp/multimap_phase6_mlp_online_smoke"
OFFLINE_DIR="${SMOKE_ROOT}/offline_mlp"
REAL_POOL="${OFFLINE_DIR}/timing_pool_real.npz"
FIXTURE_POOL="${OFFLINE_DIR}/timing_pool_functional.npz"
DATASET_DIR="${OFFLINE_DIR}/synthetic_h5"
OFFLINE_OUTPUT="${OFFLINE_DIR}/outputs"
OUTPUT_DIR="${SMOKE_ROOT}/outputs"

if [[ -e "${SMOKE_ROOT}" ]]; then
  echo "Smoke root already exists; refusing to overwrite: ${SMOKE_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${PREPARED_NPZ}" ]]; then
  echo "Required existing Phase-4 prepared NPZ is unavailable; do not rerun MIND/MP-PCA." >&2
  exit 1
fi
mkdir -p "${OFFLINE_DIR}"
cd "${PROJECT_ROOT}"
conda run -n knesvr_torch python -m modules.module_05_signal_decoder.build_timing_pool --observations "${PREPARED_NPZ}" --output "${REAL_POOL}"
conda run -n knesvr_torch python -c "from modules.module_05_signal_decoder.synthetic_dataset import make_functional_timing_fixture; make_functional_timing_fixture('${REAL_POOL}', '${FIXTURE_POOL}')"
conda run -n knesvr_torch python -m modules.module_05_signal_decoder.generate_mlp_dataset --config configs/mlp_smoke_cpu.yaml --timing-pool "${FIXTURE_POOL}" --protocol configs/protocol_hhz_v1.yaml --output-dir "${DATASET_DIR}"
conda run -n knesvr_torch python -m modules.module_05_signal_decoder.train_mlp --config configs/mlp_smoke_cpu.yaml --dataset-dir "${DATASET_DIR}" --timing-pool "${FIXTURE_POOL}" --protocol configs/protocol_hhz_v1.yaml --output-dir "${OFFLINE_OUTPUT}"
conda run -n knesvr_torch python -m scripts.run_training --config configs/recon_smoke_cpu.yaml --protocol configs/protocol_hhz_v1.yaml --observations "${PREPARED_NPZ}" --output-dir "${OUTPUT_DIR}"
