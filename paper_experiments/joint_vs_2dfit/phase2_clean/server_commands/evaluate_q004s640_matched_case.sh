#!/usr/bin/env bash
# Server-only frozen-L100 evaluation of Q004S640 versus Fit-first, Q002S640, or Q002S640_K256.
set -euo pipefail
[[ $# -eq 9 ]] || { echo "usage: $0 CONDITION ENDPOINT SPARSE_MANIFEST R0_NATIVE_MAPS_ROOT Q004_EXPORT FIT_FIRST_OR_Q002_OR_Q002S640_K256 COMPARATOR_EXPORT MASK_REFERENCE_EXPORT_ROOT FORMAL_RESULTS_ROOT" >&2; exit 2; }
CONDITION=$1; ENDPOINT=$2; MANIFEST=$(realpath -m "$3"); R0_NATIVE=$(realpath -m "$4"); Q004=$(realpath -m "$5"); KIND=$6; COMPARATOR=$(realpath -m "$7"); MASK_REFERENCE=$(realpath -m "$8"); ROOT=$(realpath -m "$9")
REPO=${REPO:-/data/dengyz/code/Code_10w}; PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}; R0_RUN_ID=${R0_RUN_ID:-R0_PHASE4R}; R1_RUN_ID=${R1_RUN_ID:-R1_PHASE4R}
case "$CONDITION" in L100_fold0|L50_fold0|L50_fold1|L25_fold0|L25_fold1|L25_fold2|L25_fold3) ;; *) echo "unsupported CYJ condition: $CONDITION" >&2; exit 2;; esac
case "$ENDPOINT" in central|psf128) ;; *) echo "unsupported endpoint: $ENDPOINT" >&2; exit 2;; esac
case "$KIND" in fit-first|q002s640|q002s640-k256) ;; *) echo "FIT_FIRST_OR_Q002_OR_Q002S640_K256 must be fit-first, q002s640, or q002s640-k256" >&2; exit 2;; esac
case "$KIND" in fit-first) LABEL="Fit-first fixed-L100";; q002s640) LABEL="Q002S640";; q002s640-k256) LABEL="Q002S640_K256";; esac
OUTPUT="$ROOT/CYJ/$CONDITION/$ENDPOINT/Q004S640_vs_${KIND}"
test -f "$MANIFEST" && test -d "$R0_NATIVE" && test -d "$Q004" && test -d "$COMPARATOR" && test -d "$MASK_REFERENCE"; test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.l100_mask_paired_eval --subject CYJ --condition "$CONDITION" --r0-run-id "$R0_RUN_ID" --r1-run-id "$R1_RUN_ID" --sparse-manifest "$MANIFEST" --r0-native-maps-root "$R0_NATIVE" --endpoint "$ENDPOINT" --proposed-export-root "$Q004" --comparator-export-root "$COMPARATOR" --mask-reference-export-root "$MASK_REFERENCE" --proposed-label Q004S640 --comparator-label "$LABEL" --output "$OUTPUT"
