#!/usr/bin/env bash
# Server-only immutable CYJ evaluation at FORMAL_RESULTS_ROOT/CYJ/CONDITION/ENDPOINT.
set -euo pipefail
[[ $# -eq 7 ]] || { echo "usage: $0 CONDITION ENDPOINT SPARSE_MANIFEST R0_NATIVE_MAPS_ROOT PROPOSED_EXPORT_ROOT FIT_FIRST_EXPORT_ROOT FORMAL_RESULTS_ROOT" >&2; exit 2; }
CONDITION=$1; ENDPOINT=$2; MANIFEST=$3; R0_NATIVE=$4; PROPOSED=$5; FIT_FIRST=$6; ROOT=$7
MANIFEST=$(realpath -m "$MANIFEST"); R0_NATIVE=$(realpath -m "$R0_NATIVE"); PROPOSED=$(realpath -m "$PROPOSED"); FIT_FIRST=$(realpath -m "$FIT_FIRST"); ROOT=$(realpath -m "$ROOT")
REPO=${REPO:-/data/dengyz/code/Code_10w}
PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
R0_RUN_ID=${R0_RUN_ID:-R0_PHASE4R}; R1_RUN_ID=${R1_RUN_ID:-R1_PHASE4R}
case "$CONDITION" in L100_fold0|L50_fold0|L50_fold1|L25_fold0|L25_fold1|L25_fold2|L25_fold3) ;; *) echo "unsupported CYJ condition: $CONDITION" >&2; exit 2;; esac
case "$ENDPOINT" in central|psf128) ;; *) echo "unsupported endpoint: $ENDPOINT" >&2; exit 2;; esac
OUTPUT="$ROOT/CYJ/$CONDITION/$ENDPOINT"
test -f "$MANIFEST" && test -d "$R0_NATIVE" && test -d "$PROPOSED" && test -d "$FIT_FIRST"
test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
printf 'CYJ matched evaluation:\n  condition=%s\n  endpoint=%s\n  manifest=%s\n  r0_native=%s\n  proposed=%s\n  fit_first=%s\n  output=%s\n' "$CONDITION" "$ENDPOINT" "$MANIFEST" "$R0_NATIVE" "$PROPOSED" "$FIT_FIRST" "$OUTPUT"
cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.l100_mask_paired_eval --subject CYJ --condition "$CONDITION" --r0-run-id "$R0_RUN_ID" --r1-run-id "$R1_RUN_ID" --sparse-manifest "$MANIFEST" --r0-native-maps-root "$R0_NATIVE" --endpoint "$ENDPOINT" --proposed-export-root "$PROPOSED" --fit-first-export-root "$FIT_FIRST" --output "$OUTPUT"
