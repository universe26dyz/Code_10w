#!/usr/bin/env bash
set -euo pipefail
CODE10W_ROOT=/data/dengyz/code/Code_10w
Q001_REFERENCE_ROOT="${Q001_REFERENCE_ROOT:?Set the migrated packaged Q001 reference root.}"
Q001_INPUT_ROOT="${Q001_INPUT_ROOT:?Set the migrated Q001 input root.}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cr_dreme
cd "${CODE10W_ROOT}"
python -m quality_experiments.Q001_recon.verify_full_fov_reference --reference-root "${Q001_REFERENCE_ROOT}" --input-root "${Q001_INPUT_ROOT}"
