#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/quality_experiments_v1/CYJ/Q001A_full_fov_full_recon_v1
[[ -f "${ROOT}/model.pt" && -f "${ROOT}/q001_route_manifest.json" ]] || { echo "Missing Q001A final checkpoint/route manifest" >&2; exit 1; }
echo "Q001A evaluation exports are produced from the final checkpoint under ${ROOT}/evaluation; compare whole full-FOV reference and exact fixed baseline ROIs only."
