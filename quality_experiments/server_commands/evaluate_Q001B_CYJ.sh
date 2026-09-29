#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/quality_experiments_v1/CYJ/Q001B_full_fov_reg_cropped_recon_v1
[[ -f "${ROOT}/model.pt" && -f "${ROOT}/q001_route_manifest.json" ]] || { echo "Missing Q001B final checkpoint/route manifest" >&2; exit 1; }
echo "Q001B evaluation uses the unchanged cropped native reference and strict B6-vs-Q001B paired support."
