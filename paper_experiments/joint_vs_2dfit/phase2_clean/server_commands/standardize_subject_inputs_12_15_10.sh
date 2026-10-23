#!/usr/bin/env bash
set -euo pipefail
mode=${1:---dry-run}
[[ "$mode" == "--dry-run" || "$mode" == "--execute" ]] || { echo "usage: $0 [--dry-run|--execute] SUBJECT PREPARED_SUBJECT_ROOT OUTPUT_ROOT" >&2; exit 2; }
shift || true
subject=${1:?usage: $0 --dry-run SUBJECT PREPARED_SUBJECT_ROOT OUTPUT_ROOT}
prepared_subject_root=${2:?usage: $0 --dry-run SUBJECT PREPARED_SUBJECT_ROOT OUTPUT_ROOT}
output_root=${3:?usage: $0 --dry-run SUBJECT PREPARED_SUBJECT_ROOT OUTPUT_ROOT}
[[ $# -eq 3 ]] || { echo "unexpected arguments" >&2; exit 2; }
python=${PYTHON:-python}
cmd=("$python" -m paper_experiments.joint_vs_2dfit.phase2_clean.standardize_subject_slices --prepared-subject-root "$prepared_subject_root" --subject "$subject" --output-root "$output_root")
printf 'dry-run only; subject=%s\nsource_root=%s\noutput_root=%s\ntargets: SAX=12 2CH=15 4CH=10\n' "$subject" "$prepared_subject_root" "$output_root"
printf 'resolved trim mapping and geometry audit (read-only):\n'
if [[ "$mode" == "--dry-run" ]]; then
    "${cmd[@]}" --dry-run
    printf 'future immutable command (not executed): '; printf '%q ' "${cmd[@]}"; printf '\n'
    exit 0
fi
"${cmd[@]}"
manifest="$output_root/$subject/slice_standardization_manifest.json"
"$python" - "$manifest" "$subject" <<'PY'
import sys
from paper_experiments.joint_vs_2dfit.phase2_clean.standardize_subject_slices import TARGET_COUNTS, validate_standardization_manifest
manifest = validate_standardization_manifest(sys.argv[1], subject=sys.argv[2])
if {stack: manifest["stacks"][stack]["output_count"] for stack in TARGET_COUNTS} != TARGET_COUNTS: raise ValueError("standardization target counts mismatch")
PY
