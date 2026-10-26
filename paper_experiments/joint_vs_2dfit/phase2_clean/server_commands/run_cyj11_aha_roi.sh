#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--dry-run" || $# -ne 1 ]]; then
  echo "usage: $0 --dry-run" >&2
  exit 2
fi

: "${DATA:?set DATA to /data/dengyz/dataset/Code_10w_v1}"
: "${REPO:?set REPO to /data/dengyz/code/Code_10w}"
: "${PYTHON:?set PYTHON to the approved server Python}"
ROOT="${DATA}/Code_10w_runs/ismrm_five_subject_analysis_v1"

cd "$REPO"
for SUBJECT in CYJ DYZ HJL HHZ DYL_2; do
  "$PYTHON" -m paper_experiments.joint_vs_2dfit.aha.run_cyj11_aha_roi \
    --mode readiness --analysis-root "$ROOT" --subject "$SUBJECT" --case L100
done
