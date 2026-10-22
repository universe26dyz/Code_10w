#!/usr/bin/env bash
set -euo pipefail
[[ ${1:---dry-run} == "--dry-run" ]] || { echo "status is read-only" >&2; exit 2; }
echo "dry-run: inspect only manifest identity, input SHA, and COMPLETE status; partial or mismatched outputs stop"
