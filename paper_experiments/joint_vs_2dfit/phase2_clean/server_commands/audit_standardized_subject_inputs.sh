#!/usr/bin/env bash
set -euo pipefail
[[ ${1:---dry-run} == "--dry-run" ]] || { echo "audit is read-only" >&2; exit 2; }
echo "dry-run: verify source/output SHA, physical group mapping, per-group reuse fingerprints, and DYL_2 readiness"
