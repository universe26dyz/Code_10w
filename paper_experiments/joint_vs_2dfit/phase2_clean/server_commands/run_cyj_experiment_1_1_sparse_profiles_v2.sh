#!/usr/bin/env bash
# Defaults to printing immutable profile commands; --execute is intentionally required.
set -euo pipefail
MODE=${1:---dry-run}
[[ "$MODE" == "--dry-run" || "$MODE" == "--execute" ]] || { echo "usage: $0 [--dry-run|--execute]" >&2; exit 2; }
[[ "$MODE" == "--dry-run" ]] || { echo "execution is deliberately delegated to an approved server session" >&2; exit 1; }
echo "dry-run: build seven CYJ profile manifests, then schedule Q002S640/Q004S640/Fit-first sequentially; no output written"
