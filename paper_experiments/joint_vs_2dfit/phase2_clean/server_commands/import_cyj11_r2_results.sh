#!/usr/bin/env bash
# Physically import completed r2 results into the unified analysis root.
set -euo pipefail

MODE=${1:---dry-run}
[[ "$MODE" == "--dry-run" || "$MODE" == "--execute" ]] || { echo "usage: $0 [--dry-run|--execute]" >&2; exit 2; }
: "${REPO:?set REPO to the checkout containing cyj11_analysis_archive}"
: "${CYJ11_R2_PYTHON:?set CYJ11_R2_PYTHON}"
: "${CYJ11_R2_SOURCE_ROOT:?set CYJ11_R2_SOURCE_ROOT to cyj11_q002_profile_v2_20261026_r2}"
: "${CYJ11_R2_ANALYSIS_ROOT:?set CYJ11_R2_ANALYSIS_ROOT to ismrm_five_subject_analysis_v1}"
: "${CYJ11_R2_SUBJECT:?set CYJ11_R2_SUBJECT to DYZ, HJL, HHZ, or DYL_2}"

CASES=(SAX100_2CH0_4CH0 SAX100_2CH0_4CH100 SAX100_2CH100_4CH0 S2CH50_4CH100_fold0 S2CH50_4CH100_fold1 S2CH100_4CH50_fold0 S2CH100_4CH50_fold1)
for CASE in "${CASES[@]}"; do
    [[ -d "$CYJ11_R2_SOURCE_ROOT/$CYJ11_R2_SUBJECT/$CASE" ]] || { echo "missing r2 case: $CASE" >&2; exit 1; }
done

if [[ "$MODE" == "--dry-run" ]]; then
    printf 'r2 import subject=%s cases=%s\n' "$CYJ11_R2_SUBJECT" "${CASES[*]}"
    exit 0
fi

(cd "$REPO" && "$CYJ11_R2_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_analysis_archive import-r2 --r2-root "$CYJ11_R2_SOURCE_ROOT" --analysis-root "$CYJ11_R2_ANALYSIS_ROOT" --subject "$CYJ11_R2_SUBJECT")
