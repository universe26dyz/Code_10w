#!/usr/bin/env bash
set -euo pipefail
# Server-only formal runner; never invoked automatically by Codex.
RUN_ID=${1:?usage: run_real_full_fit_first.sh RUN_ID}; ROOT=${PHASE4R_RUN_ROOT:-/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/paper_joint_vs_2dfit/real/R0_full}
REPO=${REPO:-/data/dengyz/code/Code_10w}; DATA=${DATA:-/data/dengyz/dataset/Code_10w_v1}; PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
export PHASE4R_PREFLIGHT_JSON=${PHASE4R_PREFLIGHT_JSON:-"$ROOT/$RUN_ID/preflight_fit_first.json"}; "$(dirname "$0")/server_preflight_real.sh" >/dev/null
test ! -e "$ROOT/$RUN_ID/fit_first" || { echo "refusing overwrite: $ROOT/$RUN_ID/fit_first" >&2; exit 1; }; cd "$REPO"
mkdir -p "$ROOT/$RUN_ID/fit_first"
NESVOR=$($PY -c 'import json,os; print(json.load(open(os.environ["PHASE4R_PREFLIGHT_JSON"]))["historical_nesvor"]["executable"])')
for SID in CYJ DYZ HHZ HJL; do
  OUT="$ROOT/$RUN_ID/fit_first/$SID"; "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.baseline --prepared-root "$DATA/Code_10w_prepared" --subject-id "$SID" --nesvor-executable "$NESVOR" --output "$OUT" --execute 2>&1 | tee "$OUT.log"; "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.run_provenance --output "$OUT" --method fit_first --subject "$SID" --prepared-root "$DATA/Code_10w_prepared" --nesvor-executable "$NESVOR"
done
