#!/usr/bin/env bash
set -euo pipefail
REPO=/data/dengyz/code/Code_10w; DATA=/data/dengyz/dataset/Code_10w_v1; EXPECTED=f83280d4f73bfc9e0540b9606ee20bf83395339ad4e54dd0fb7be93ed9c9a7a9
cd "$REPO"; test -z "$(git status --porcelain)" || { echo 'refusing dirty git'; exit 1; }; test -f paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_1_clean_mrxcat/phase1_final_gate_matlab_parity.json
mapfile -t CANDIDATES < <(find "$DATA/Code_10w_runs" -type f -name '*.pth' -print0 | xargs -0 -r sha256sum | awk -v h="$EXPECTED" '$1==h {print $2}')
test "${#CANDIDATES[@]}" -eq 1 || { echo "expected exactly one approved decoder by SHA256; copy the approved checkpoint if zero"; exit 1; }
python - <<'PY'
import json
p='paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_1_clean_mrxcat/phase1_final_gate_matlab_parity.json'
d=json.load(open(p)); assert d['formal_synthetic_reconstruction_ready'] is True
PY
echo "approved_decoder=${CANDIDATES[0]}"; git rev-parse HEAD
