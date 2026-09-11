#!/usr/bin/env bash
set -euo pipefail

: "${PREPARED_NPZ:?set PREPARED_NPZ to one prepared observations.npz path}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to an absent or empty output directory}"
cd /home/universe/SVR/multimap_postprogramming/Code_10w/trad
python -m scripts.run_training \
  --config configs/server_train_example.yaml \
  --protocol configs/protocol_hhz_v1.yaml \
  --observations "${PREPARED_NPZ}" \
  --output-dir "${OUTPUT_DIR}"
