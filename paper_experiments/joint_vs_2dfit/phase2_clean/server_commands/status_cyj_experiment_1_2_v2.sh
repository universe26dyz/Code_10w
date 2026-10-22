#!/usr/bin/env bash
set -euo pipefail
[[ ${1:---dry-run} == "--dry-run" ]] || { echo "status is read-only" >&2; exit 2; }
echo "read-only preflight: inspect Q005 manifest, decoder/B6 SHA256, K=4/2560/chunk=4 controls, and distinct central/psf128 endpoint provenance"
