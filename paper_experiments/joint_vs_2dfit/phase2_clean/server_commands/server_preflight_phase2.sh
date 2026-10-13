#!/usr/bin/env bash
set -euo pipefail
REPO=${REPO:-/data/dengyz/code/Code_10w}; DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}; EXPECTED=f83280d4f73bfc9e0540b9606ee20bf83395339ad4e54dd0fb7be93ed9c9a7a9
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cr_dreme; cd "$REPO"
test -z "$(git status --porcelain)" || { echo 'refusing dirty git' >&2; exit 1; }
python -c 'import torch; assert torch.cuda.is_available(); import tinycudann; import nesvor; import paper_experiments.joint_vs_2dfit.phase2_clean.runner'
python - <<'PY'
import json
p='paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_1_clean_mrxcat/phase1_final_gate_matlab_parity.json'; assert json.load(open(p))['formal_synthetic_reconstruction_ready'] is True
PY
for sid in CYJ DYZ HHZ HJL; do for stack in sax 2ch 4ch; do test -f "$DATA/Code_10w_prepared/$sid/$stack/observations.npz"; done; done
mapfile -t CANDIDATES < <(find "$DATA/Code_10w_runs" -type f -name '*.pth' -print0 | xargs -0 -r sha256sum | awk -v h="$EXPECTED" '$1==h {print $2}')
test "${#CANDIDATES[@]}" -eq 1 || { echo "expected exactly one approved decoder by SHA256; run: find /data/dengyz/dataset/Code_10w_v1/Code_10w_runs -type f -name '*.pth' -print0 | xargs -0 -r sha256sum | grep '^$EXPECTED '" >&2; exit 1; }
B6="$DATA/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/runs/frozen_mlp_B6/model.pt"; test -f "$B6"
RESULT=${PHASE2_PREFLIGHT_JSON:-/tmp/phase2_preflight_actual.json}; rm -f "$RESULT"
python -m paper_experiments.joint_vs_2dfit.phase2_clean.preflight_actual --manifest paper_experiments/joint_vs_2dfit/phase2_clean/phase2_input_manifest.json --reference-manifest paper_experiments/joint_vs_2dfit/phase2_clean/final_real_reference_manifest.json --data-root "$DATA/Code_10w_prepared" --synthetic-root paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_1_clean_mrxcat --decoder "${CANDIDATES[0]}" --b6-model "$B6" --output "$RESULT" >/dev/null
python - "$RESULT" <<'PY'
import json, subprocess, sys
d=json.load(open(sys.argv[1])); d['code_head']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(); print(json.dumps(d))
PY
