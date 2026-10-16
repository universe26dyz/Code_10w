#!/usr/bin/env bash
set -euo pipefail
# Server-only gate for Phase-4R real experiments.  It deliberately never reads XCAT/synthetic artifacts.
REPO=${REPO:-/data/dengyz/code/Code_10w}; DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}
CR_PYTHON=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
NESVOR_EXEC=${PHASE2_NESVOR_EXECUTABLE:-/home/dengyz/miniconda3/envs/nesvor/bin/nesvor}
NESVOR_PYTHON=${PHASE2_NESVOR_PYTHON:-/home/dengyz/miniconda3/envs/nesvor/bin/python}
NESVOR_SOURCE=${PHASE2_NESVOR_SOURCE:-/data/dengyz/code/NeSVoR-master/nesvor}
OUTPUT=${PHASE4R_PREFLIGHT_JSON:-/tmp/phase4r_real_preflight_actual.json}
test -x "$CR_PYTHON"; cd "$REPO"
test -z "$(git status --porcelain --untracked-files=no)" || { echo 'refusing dirty tracked git worktree' >&2; exit 1; }
"$CR_PYTHON" -c 'import torch, tinycudann; raise SystemExit(0 if torch.cuda.is_available() else "CUDA unavailable")'
test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
RELATIVE=$($CR_PYTHON -c 'from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import CANONICAL_APPROVED_DECODER_RELATIVE_PATH; print(CANONICAL_APPROVED_DECODER_RELATIVE_PATH)')
"$CR_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.preflight_real \
  --manifest paper_experiments/joint_vs_2dfit/phase2_clean/phase2_input_manifest.json \
  --reference-manifest paper_experiments/joint_vs_2dfit/phase2_clean/final_real_reference_manifest.json \
  --reference-root "$DATA" --data-root "$DATA/Code_10w_prepared" \
  --decoder "${PHASE2_APPROVED_DECODER:-$DATA/$RELATIVE}" \
  --approved-decoder-audit paper_experiments/joint_vs_2dfit/audit/outputs/approved_decoder.json \
  --b6-model "$DATA/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/runs/frozen_mlp_B6/model.pt" \
  --nesvor-executable "$NESVOR_EXEC" --nesvor-python "$NESVOR_PYTHON" --nesvor-source "$NESVOR_SOURCE" --output "$OUTPUT" >/dev/null
cat "$OUTPUT"
