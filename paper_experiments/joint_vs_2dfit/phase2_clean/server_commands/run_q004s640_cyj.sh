#!/usr/bin/env bash
# Server-only Q004S640 K1 reconstruction through the phase2 subject case adapter.
set -euo pipefail
[[ $# -eq 7 ]] || { echo "usage: $0 CONDITION L100_R0_PREPARED_ROOT SPARSE_PREPARED_ROOT SPARSE_MANIFEST B6_MODEL APPROVED_DECODER NEW_OUTPUT" >&2; exit 2; }
CONDITION=$1; L100=$(realpath -m "$2"); SPARSE=$(realpath -m "$3"); MANIFEST=$(realpath -m "$4"); B6=$(realpath -m "$5"); DECODER=$(realpath -m "$6"); OUTPUT=$(realpath -m "$7")
REPO=${REPO:-/data/dengyz/code/Code_10w}; PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}; SUBJECT=${SUBJECT:-CYJ}
case "$CONDITION" in L100_fold0|L50_fold0|L50_fold1|L25_fold0|L25_fold1|L25_fold2|L25_fold3) ;; *) echo "unsupported formal condition: $CONDITION" >&2; exit 2;; esac
test -d "$L100" && test -d "$SPARSE" && test -f "$MANIFEST" && test -f "$B6" && test -f "$DECODER"
test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.q004s640 --subject "$SUBJECT" --condition "$CONDITION" --l100-prepared-root "$L100" --sparse-prepared-root "$SPARSE" --sparse-manifest "$MANIFEST" --b6-model "$B6" --decoder "$DECODER" --output "$OUTPUT" --device cuda:0 --execute
