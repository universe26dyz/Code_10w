#!/usr/bin/env bash
set -euo pipefail

# Read-only recorder. Run this on the server before copying a live route.
# It never writes under CODE_ROOT or RUN_ROOT.
CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to the current corrected code checkout}
ARCHIVE_ROOT=${ARCHIVE_ROOT:?set ARCHIVE_ROOT to ismrm_v2_1}
RUNS_ROOT=${RUNS_ROOT:?set RUNS_ROOT to the corrected run parent}
DECODER=${DECODER:?set active [10,100] decoder path}
B6_MODEL=${B6_MODEL:?set B6 model path}
PROTOCOL=${PROTOCOL:?set corrected protocol path}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$ARCHIVE_ROOT/runtime/source_records/live_$STAMP"
mkdir -p "$OUT"

git -C "$CODE_ROOT" rev-parse HEAD > "$OUT/git_head.txt"
git -C "$CODE_ROOT" status --short > "$OUT/git_status.txt"
sha256sum "$DECODER" "$B6_MODEL" "$PROTOCOL" > "$OUT/model_protocol_sha256.txt"
printf '%s\n' HHZ CYJ DYZ DYL_2 HJL | while read -r subject; do
  run="$RUNS_ROOT/$subject"
  if [ -d "$run" ]; then
    printf '{"subject":"%s","run_root":"%s","exists":true}\n' "$subject" "$run"
  else
    printf '{"subject":"%s","run_root":"%s","exists":false}\n' "$subject" "$run"
  fi
done > "$OUT/live_run_contract.jsonl"
find "$CODE_ROOT" -type f -not -path '*/.git/*' -print0 | sort -z | xargs -0 sha256sum > "$OUT/source_file_manifest.sha256"
printf 'LIVE SOURCE RECORD: %s\n' "$OUT"

