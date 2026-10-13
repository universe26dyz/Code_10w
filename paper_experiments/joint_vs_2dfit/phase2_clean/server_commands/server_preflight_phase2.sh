#!/usr/bin/env bash
set -euo pipefail
REPO=${REPO:-/data/dengyz/code/Code_10w}; DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}; EXPECTED=f83280d4f73bfc9e0540b9606ee20bf83395339ad4e54dd0fb7be93ed9c9a7a9
NESVOR_EXEC=${PHASE2_NESVOR_EXECUTABLE:-/home/dengyz/miniconda3/envs/nesvor/bin/nesvor}; NESVOR_PYTHON=${PHASE2_NESVOR_PYTHON:-/home/dengyz/miniconda3/envs/nesvor/bin/python}; NESVOR_SOURCE=${PHASE2_NESVOR_SOURCE:-/data/dengyz/code/NeSVoR-master/nesvor}
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cr_dreme; cd "$REPO"
test -z "$(git status --porcelain)" || { echo 'refusing dirty git' >&2; exit 1; }
python -c 'import torch, tinycudann, paper_experiments.joint_vs_2dfit.phase2_clean.runner; raise SystemExit(0 if torch.cuda.is_available() else "CUDA unavailable")'
NESVOR_RUNTIME_JSON=$("$NESVOR_PYTHON" "$REPO/paper_experiments/joint_vs_2dfit/phase2_clean/historical_nesvor_runtime.py" --executable "$NESVOR_EXEC" --python "$NESVOR_PYTHON" --source-root "$NESVOR_SOURCE")
python - <<'PY'
import json
p='paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_1_clean_mrxcat/phase1_final_gate_matlab_parity.json'
if json.load(open(p)).get('formal_synthetic_reconstruction_ready') is not True: raise SystemExit('Phase-1 final gate is not ready')
PY
for sid in CYJ DYZ HHZ HJL; do for stack in sax 2ch 4ch; do test -f "$DATA/Code_10w_prepared/$sid/$stack/observations.npz"; done; done
mapfile -t CANDIDATES < <(find "$DATA/Code_10w_runs" -type f -print0 | xargs -0 -r sha256sum | awk -v h="$EXPECTED" '$1==h {print $2}')
test "${#CANDIDATES[@]}" -eq 1 || { echo "expected exactly one approved decoder by SHA256; run: find /data/dengyz/dataset/Code_10w_v1/Code_10w_runs -type f -print0 | xargs -0 -r sha256sum | grep '^$EXPECTED '" >&2; exit 1; }
B6="$DATA/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/runs/frozen_mlp_B6/model.pt"; test -f "$B6"
RESULT=${PHASE2_PREFLIGHT_JSON:-/tmp/phase2_preflight_actual.json}; rm -f "$RESULT"
python -m paper_experiments.joint_vs_2dfit.phase2_clean.preflight_actual --manifest paper_experiments/joint_vs_2dfit/phase2_clean/phase2_input_manifest.json --reference-manifest paper_experiments/joint_vs_2dfit/phase2_clean/final_real_reference_manifest.json --reference-root "$DATA" --data-root "$DATA/Code_10w_prepared" --synthetic-root paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_1_clean_mrxcat --decoder "${CANDIDATES[0]}" --b6-model "$B6" --output "$RESULT" >/dev/null
python - "$RESULT" "$NESVOR_RUNTIME_JSON" <<'PY'
import json, subprocess, sys
d=json.load(open(sys.argv[1])); d.update(json.loads(sys.argv[2])); d['code_head']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(); d['environment_gate']=True; d['phase1_gate']=True; d['formal_reconstruction_ready']=all(d.get(key) is True for key in ('actual_observation_hash_gate','references_ready','approved_decoder_gate','b6_config_source_gate','historical_nesvor_runtime_gate','environment_gate','phase1_gate')); print(json.dumps(d))
PY
