#!/usr/bin/env bash
set -euo pipefail
CODE10W_ROOT=/data/dengyz/code/Code_10w
Q001_INPUT_ROOT="${Q001_INPUT_ROOT:?Set the migrated Q001 full-FOV input root.}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cr_dreme
cd "${CODE10W_ROOT}"
python -c 'import torch; assert torch.cuda.is_available(), "Q001 server verification requires the formal GPU environment"'
python -m quality_experiments.Q001_recon.verify_inputs --input-root "${Q001_INPUT_ROOT}"
