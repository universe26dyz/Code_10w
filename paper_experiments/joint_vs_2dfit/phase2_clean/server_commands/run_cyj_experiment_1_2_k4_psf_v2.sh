#!/usr/bin/env bash
set -euo pipefail
[[ ${1:---dry-run} == "--dry-run" ]] || { echo "K4 execution requires separate post-1.1 user approval" >&2; exit 1; }
echo "dry-run: Q005S640_K4 on untrimmed CYJ L100 only; training_psf_samples=4; central and psf128 are post-hoc endpoints"
