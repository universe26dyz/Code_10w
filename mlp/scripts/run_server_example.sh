#!/usr/bin/env bash
set -euo pipefail

: "${PREPARED_SAX_NPZ:?set PREPARED_SAX_NPZ explicitly}"
: "${PREPARED_2CH_NPZ:?set PREPARED_2CH_NPZ explicitly}"
: "${PREPARED_4CH_NPZ:?set PREPARED_4CH_NPZ explicitly}"
: "${MLP_CHECKPOINT:?set MLP_CHECKPOINT explicitly}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to an absent or empty output directory}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${OUTPUT_DIR}.recon_config.yaml"
if [[ -e "${CONFIG}" ]]; then
  echo "Refusing to overwrite generated server config: ${CONFIG}" >&2
  exit 1
fi
cd "${PROJECT_ROOT}"
MLP_CHECKPOINT="${MLP_CHECKPOINT}" CONFIG="${CONFIG}" conda run -n cr_dreme python -c "import os, pathlib, yaml; source=pathlib.Path('configs/recon_server_train_example.yaml'); cfg=yaml.safe_load(source.read_text()); cfg['decoder']['checkpoint']=os.environ['MLP_CHECKPOINT']; pathlib.Path(os.environ['CONFIG']).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')"
conda run -n cr_dreme python -m scripts.run_training \
  --config "${CONFIG}" --protocol configs/protocol_hhz_v1.yaml \
  --observations "${PREPARED_SAX_NPZ}" --observations "${PREPARED_2CH_NPZ}" --observations "${PREPARED_4CH_NPZ}" \
  --output-dir "${OUTPUT_DIR}"
