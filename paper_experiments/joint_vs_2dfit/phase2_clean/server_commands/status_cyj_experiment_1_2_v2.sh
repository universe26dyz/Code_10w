#!/usr/bin/env bash
set -euo pipefail
[[ ${1:---dry-run} == "--dry-run" ]] || { echo "status is read-only" >&2; exit 2; }
echo "dry-run: inspect only Q005 manifest, decoder/B6 SHA, K4 control identity, and endpoint provenance"
