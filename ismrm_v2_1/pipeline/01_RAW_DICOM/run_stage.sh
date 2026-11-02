#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SUBJECT=${1:?usage: $0 SUBJECT RAW_ROOT}
RAW_ROOT=${2:?usage: $0 SUBJECT RAW_ROOT}
MANIFEST="$PROJECT_ROOT/subjects/$SUBJECT/manifest.json"
mkdir -p "$(dirname "$MANIFEST")"
python "$PROJECT_ROOT/common/scripts/extract_subject_deployment_manifest.py" \
  --source "$PROJECT_ROOT/configs/subject_stack_manifest.example.json" \
  --subject "$SUBJECT" --output "$MANIFEST"
for stack in sax 2ch 4ch; do
  count=$(find "$RAW_ROOT/$SUBJECT/$stack" -maxdepth 1 -type f -name '*.dcm' | wc -l)
  [ "$count" -gt 0 ] && [ $((count % 10)) -eq 0 ] || { echo "invalid $stack DICOM count: $count" >&2; exit 1; }
  echo "$stack DICOM files: $count"
done
echo "RAW DICOM: PASS"

