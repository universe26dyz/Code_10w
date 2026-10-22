#!/usr/bin/env bash
set -euo pipefail
[[ ${1:---dry-run} == "--dry-run" ]] || { echo "standardization execution must be separately approved" >&2; exit 1; }
echo "dry-run: immutable physical-order standardization to SAX=12,2CH=15,4CH=10; no prepared input is modified"
