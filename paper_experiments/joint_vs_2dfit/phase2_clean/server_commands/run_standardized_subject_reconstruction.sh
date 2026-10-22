#!/usr/bin/env bash
set -euo pipefail
[[ ${1:---dry-run} == "--dry-run" ]] || { echo "reconstruction execution must be separately approved" >&2; exit 1; }
echo "dry-run: fresh Proposed, Fit-first T1/T2, poses, endpoints, bundles, masks, metrics, figures, and summaries; no old 3D result reuse"
