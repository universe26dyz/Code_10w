#!/usr/bin/env bash
set -euo pipefail
[[ ${1:---dry-run} == "--dry-run" ]] || { echo "reconstruction execution is not authorized by this wrapper" >&2; exit 2; }
cat <<'PLAN'
dry-run only: no GPU, NeSVoR, training, endpoint export, metrics, or figures will run.
Required preflight: verified 12/15/10 standardization manifest; REUSE_VERIFIED maps;
frozen decoder/B6 SHA256; target-server availability; fresh empty output root.
Future Proposed command: python -m quality_experiments.Q002_B6_roi_same_anchor_mse.run_q002_reconstruction --experiment-id Q002S640_B6_roi_same_anchor_mse ...
Future Fit-first T1/T2 command: nesvor reconstruct --input-stacks ... (independent parameter runs)
Then regenerate: poses, central/psf128 endpoints, native bundles, support masks, metrics, figures, summary.
Old 3D reconstructions and their downstream artifacts are never reused.
PLAN
