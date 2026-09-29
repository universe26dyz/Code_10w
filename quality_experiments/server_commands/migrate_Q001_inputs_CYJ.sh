#!/usr/bin/env bash
set -euo pipefail
LOCAL_Q001_INPUT_ROOT=/home/universe/SVR/data/Code_10w_v1/Q001_full_fov_inputs_v2
SERVER_HOST="${Q001_SERVER_HOST:?Set the formal server host.}"
SERVER_Q001_INPUT_ROOT=/data/dengyz/dataset/Code_10w_v1/Q001_full_fov_inputs_v2
[[ -f "${LOCAL_Q001_INPUT_ROOT}/q001_input_manifest.json" ]] || { echo "Missing local ready Q001 input manifest" >&2; exit 1; }
ssh "${SERVER_HOST}" "test ! -e '${SERVER_Q001_INPUT_ROOT}' || test -z \"\$(find '${SERVER_Q001_INPUT_ROOT}' -mindepth 1 -maxdepth 1 -print -quit)\"" || { echo "Refusing non-empty remote Q001 destination: ${SERVER_Q001_INPUT_ROOT}" >&2; exit 1; }
rsync -a --checksum --protect-args "${LOCAL_Q001_INPUT_ROOT}/" "${SERVER_HOST}:${SERVER_Q001_INPUT_ROOT}/"
printf 'Server verification: Q001_INPUT_ROOT=%q bash quality_experiments/server_commands/verify_Q001_inputs_CYJ.sh\n' "${SERVER_Q001_INPUT_ROOT}"
