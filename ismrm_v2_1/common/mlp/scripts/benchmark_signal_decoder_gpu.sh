#!/usr/bin/env bash
set -euo pipefail

: "${MLP_CHECKPOINT:?set MLP_CHECKPOINT explicitly}"
: "${RR_DATASET_DIR:?set RR_DATASET_DIR explicitly}"
: "${BENCHMARK_OUTPUT:?set BENCHMARK_OUTPUT explicitly}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
conda run --no-capture-output -n knesvr_torch python -m mlp.modules.module_09_qc_benchmark.benchmark_signal_decoder \
  --checkpoint "${MLP_CHECKPOINT}" --rr-dataset-dir "${RR_DATASET_DIR}" --protocol mlp/configs/protocol_hhz_v1.yaml \
  --output "${BENCHMARK_OUTPUT}" --device cuda:0 --batch-size 1000 --batch-size 10000 --repetitions 3
