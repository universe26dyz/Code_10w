#!/usr/bin/env bash
set -euo pipefail

: "${MLP_CHECKPOINT:?set MLP_CHECKPOINT explicitly}"
: "${TIMING_POOL:?set TIMING_POOL explicitly}"
: "${BENCHMARK_OUTPUT:?set BENCHMARK_OUTPUT explicitly}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
METHOD_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${METHOD_ROOT}"
conda run -n cr_dreme python -m modules.module_09_qc_benchmark.benchmark_signal_decoder \
  --checkpoint "${MLP_CHECKPOINT}" --timing-pool "${TIMING_POOL}" --protocol configs/protocol_hhz_v1.yaml \
  --output "${BENCHMARK_OUTPUT}" --device cuda:0 --batch-size 1000 --batch-size 10000 --repetitions 3
