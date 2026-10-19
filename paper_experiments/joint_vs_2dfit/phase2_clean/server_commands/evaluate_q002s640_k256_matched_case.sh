#!/usr/bin/env bash
# Server-only frozen-L100 evaluator: Q002S640_K256 versus FIT_FIRST_OR_Q002S640 at one matched endpoint.
set -euo pipefail
[[ $# -eq 8 ]] || { echo "usage: $0 CONDITION ENDPOINT SPARSE_MANIFEST R0_NATIVE_MAPS_ROOT K256_EXPORT FIT_FIRST_OR_Q002S640 COMPARATOR_EXPORT FORMAL_RESULTS_ROOT" >&2; exit 2; }
CONDITION=$1; ENDPOINT=$2; MANIFEST=$(realpath -m "$3"); R0_NATIVE=$(realpath -m "$4"); K256=$(realpath -m "$5"); COMPARATOR_KIND=$6; COMPARATOR=$(realpath -m "$7"); ROOT=$(realpath -m "$8")
REPO=${REPO:-/data/dengyz/code/Code_10w}; PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
R0_RUN_ID=${R0_RUN_ID:-R0_PHASE4R}; R1_RUN_ID=${R1_RUN_ID:-R1_PHASE4R}
case "$CONDITION" in L100_fold0|L50_fold0|L50_fold1|L25_fold0|L25_fold1|L25_fold2|L25_fold3) ;; *) echo "unsupported CYJ condition: $CONDITION" >&2; exit 2;; esac
case "$ENDPOINT" in central|psf128) ;; *) echo "unsupported endpoint: $ENDPOINT" >&2; exit 2;; esac
case "$COMPARATOR_KIND" in fit-first|q002s640) ;; *) echo "FIT_FIRST_OR_Q002S640 must be fit-first or q002s640" >&2; exit 2;; esac
OUTPUT="$ROOT/CYJ/$CONDITION/$ENDPOINT/Q002S640_K256_vs_${COMPARATOR_KIND}"
test -f "$MANIFEST" && test -d "$R0_NATIVE" && test -d "$K256" && test -d "$COMPARATOR"
test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.l100_mask_paired_eval --subject CYJ --condition "$CONDITION" --r0-run-id "$R0_RUN_ID" --r1-run-id "$R1_RUN_ID" --sparse-manifest "$MANIFEST" --r0-native-maps-root "$R0_NATIVE" --endpoint "$ENDPOINT" --proposed-export-root "$K256" --fit-first-export-root "$COMPARATOR" --output "$OUTPUT"
