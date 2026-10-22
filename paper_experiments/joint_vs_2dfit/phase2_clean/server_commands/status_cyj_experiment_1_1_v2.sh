#!/usr/bin/env bash
set -euo pipefail
[[ ${1:---dry-run} == "--dry-run" ]] || { echo "status is read-only" >&2; exit 2; }
echo "read-only preflight: profile_id/case_id, active-stack paths, source SHA256, held-out IDs, and endpoint provenance must be checked; any missing evidence blocks launch"
