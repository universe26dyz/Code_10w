#!/usr/bin/env bash
# Server-only post-hoc endpoint aliases for an existing Q002S640_K256 checkpoint; it never trains or reconstructs.
set -euo pipefail
[[ $# -eq 4 ]] || { echo "usage: $0 CONDITION Q002S640_K256_SOURCE_ROOT CENTRAL_NEW_OUTPUT PSF128_NEW_OUTPUT" >&2; exit 2; }
CONDITION=$1; SOURCE=$(realpath -m "$2"); CENTRAL=$(realpath -m "$3"); PSF128=$(realpath -m "$4")
REPO=${REPO:-/data/dengyz/code/Code_10w}; PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
case "$CONDITION" in L100_fold0|L50_fold0|L50_fold1|L25_fold0|L25_fold1|L25_fold2|L25_fold3) ;; *) echo "unsupported CYJ condition: $CONDITION" >&2; exit 2;; esac
test -d "$SOURCE"
test ! -e "$CENTRAL" || { echo "refusing overwrite: $CENTRAL" >&2; exit 1; }
test ! -e "$PSF128" || { echo "refusing overwrite: $PSF128" >&2; exit 1; }
cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj_matched_comparison alias-q002s640-k256 --condition "$CONDITION" --source-root "$SOURCE" --endpoint central --output "$CENTRAL"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj_matched_comparison alias-q002s640-k256 --condition "$CONDITION" --source-root "$SOURCE" --endpoint psf128 --output "$PSF128"
