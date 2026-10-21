#!/usr/bin/env bash
# Server-only reporting/figure wrapper. It reads completed formal exports only.
set -euo pipefail

SUBJECT=${SUBJECT:-CYJ}
if [[ ${1:-} == "--subject" ]]; then
  [[ $# -ge 3 && -n ${2:-} ]] || { echo "--subject requires an explicit identity" >&2; exit 2; }
  SUBJECT=$2
  shift 2
fi
[[ $# -ge 1 ]] || { echo "usage: $0 [--subject SUBJECT] aggregate NEW_SUMMARY_ROOT FIT_FIRST_SOURCE_METHOD METHOD=FORMAL_RESULT_ROOT [METHOD=FORMAL_RESULT_ROOT ...] | $0 [--subject SUBJECT] figures FIGURE_CONFIG_JSON NEW_FIGURE_ROOT" >&2; exit 2; }
MODE=$1
shift
REPO=${REPO:-/data/dengyz/code/Code_10w}
PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}

case "$MODE" in
  aggregate)
    [[ $# -ge 3 ]] || { echo "aggregate requires NEW_SUMMARY_ROOT FIT_FIRST_SOURCE_METHOD and one or more METHOD=FORMAL_RESULT_ROOT values" >&2; exit 2; }
    OUTPUT=$(realpath -m "$1")
    FIT_FIRST_SOURCE_METHOD=$2
    shift 2
    test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
    ARGS=(aggregate --subject "$SUBJECT" --fit-first-source-method "$FIT_FIRST_SOURCE_METHOD" --output "$OUTPUT")
    for ITEM in "$@"; do
      [[ "$ITEM" == *=* ]] || { echo "method root must be METHOD=FORMAL_RESULT_ROOT: $ITEM" >&2; exit 2; }
      ROOT=${ITEM#*=}
      test -d "$ROOT" || { echo "formal result root is missing: $ROOT" >&2; exit 1; }
      ARGS+=(--method-root "$ITEM")
    done
    ;;
  figures)
    [[ $# -eq 2 ]] || { echo "figures requires FIGURE_CONFIG_JSON NEW_FIGURE_ROOT" >&2; exit 2; }
    CONFIG=$(realpath -m "$1")
    OUTPUT=$(realpath -m "$2")
    test -f "$CONFIG"
    test ! -e "$OUTPUT" || { echo "refusing overwrite: $OUTPUT" >&2; exit 1; }
    ARGS=(figures --config "$CONFIG" --output "$OUTPUT")
    ;;
  *)
    echo "mode must be aggregate or figures: $MODE" >&2
    exit 2
    ;;
esac

cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.final_cyj_reporting "${ARGS[@]}"
