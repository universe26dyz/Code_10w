#!/usr/bin/env bash
set -euo pipefail

: "${PREPARED_SAX_NPZ:?set PREPARED_SAX_NPZ to prepared SAX observations.npz}"
: "${PREPARED_2CH_NPZ:?set PREPARED_2CH_NPZ to prepared 2CH observations.npz}"
: "${PREPARED_4CH_NPZ:?set PREPARED_4CH_NPZ to prepared 4CH observations.npz}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to an absent or empty output directory}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
METHOD_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${METHOD_ROOT}"
conda run -n cr_dreme python -m scripts.run_training \
  --config configs/server_train_example.yaml \
  --protocol configs/protocol_hhz_v1.yaml \
  --observations "${PREPARED_SAX_NPZ}" \
  --observations "${PREPARED_2CH_NPZ}" \
  --observations "${PREPARED_4CH_NPZ}" \
  --output-dir "${OUTPUT_DIR}"
