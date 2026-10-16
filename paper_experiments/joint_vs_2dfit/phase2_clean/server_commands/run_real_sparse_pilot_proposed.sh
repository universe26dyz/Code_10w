#!/usr/bin/env bash
set -euo pipefail
# Server-only CYJ sparse runner.  L100 is intentionally reused from R0, never reconstructed again.
[[ $# -eq 3 ]] || { echo "usage: $0 RUN_ID CONDITION PHASE" >&2; exit 2; }
RUN_ID=$1; CONDITION=$2; PHASE=$3; test "$CONDITION" != L100 || { echo 'L100 reuses R0; no sparse reconstruction is allowed' >&2; exit 2; }
DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}; REPO=${REPO:-/data/dengyz/code/Code_10w}; PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
INPUT="$DATA/Code_10w_runs/paper_joint_vs_2dfit/sparse_inputs/CYJ/${CONDITION}_fold${PHASE}"; OUT="$DATA/Code_10w_runs/paper_joint_vs_2dfit/real/R1_sparse/$RUN_ID/proposed/CYJ/${CONDITION}_fold${PHASE}"
test -f "$INPUT/sparse_manifest.json"; test ! -e "$OUT" || { echo "refusing overwrite: $OUT" >&2; exit 1; }; cd "$REPO"
DECODER=${PHASE2_APPROVED_DECODER:-$DATA/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/inputs/mlp/signal_simulator_approved.pth}
B6="$DATA/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/runs/frozen_mlp_B6/model.pt"; "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.proposed --prepared-root "$INPUT" --subject-id CYJ --decoder "$DECODER" --b6-model "$B6" --output "$OUT" --execute
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.run_provenance --output "$OUT" --method proposed --subject CYJ --prepared-root "$INPUT" --decoder "$DECODER" --b6-model "$B6"
