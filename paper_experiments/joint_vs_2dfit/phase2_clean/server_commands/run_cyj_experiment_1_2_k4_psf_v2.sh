#!/usr/bin/env bash
set -euo pipefail
[[ ${1:---dry-run} == "--dry-run" ]] || { echo "K4 execution is not authorized" >&2; exit 2; }
echo "dry-run: future Q005S640_K4 uses 640 anchors x K4=2560 locations, psf_forward_chunk_size=4; central and psf128 remain post-hoc endpoints; no command is executed"
