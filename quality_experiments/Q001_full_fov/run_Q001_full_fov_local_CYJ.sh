#!/usr/bin/env bash
set -euo pipefail

CODE10W_ROOT=/home/universe/SVR/multimap_postprogramming/Code_10w
MANIFEST="${CODE10W_ROOT}/configs/subject_stack_manifest_CYJ_done.json"
Q001_OUTPUT_ROOT="${Q001_OUTPUT_ROOT:?Set an explicit new Q001 output root; never use baseline inputs/preprocessed or inputs/prepared.}"

[[ -f "${MANIFEST}" ]] || { echo "Missing deployment manifest: ${MANIFEST}" >&2; exit 1; }
if [[ -e "${Q001_OUTPUT_ROOT}" ]] && [[ -n "$(find "${Q001_OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty Q001 output root: ${Q001_OUTPUT_ROOT}" >&2; exit 1
fi
mkdir -p "${Q001_OUTPUT_ROOT}/full_fov_preprocessed/CYJ" "${Q001_OUTPUT_ROOT}/full_fov_prepared/CYJ"
cd "${CODE10W_ROOT}"

for stack in sax 2ch 4ch; do
  DICOM_DIR="$(conda run --no-capture-output -n knesvr_torch python -c "from scripts.deployment_manifest import load_deployment_manifest; print(next(x['local_dicom_dir'] for x in load_deployment_manifest('${MANIFEST}') if x['subject_id']=='CYJ' and x['stack']=='${stack}'))")"
  MAT_PATH="${Q001_OUTPUT_ROOT}/full_fov_preprocessed/CYJ/${stack}/preprocessed.mat"
  PREPARED_DIR="${Q001_OUTPUT_ROOT}/full_fov_prepared/CYJ/${stack}"
  [[ -d "${DICOM_DIR}" ]] || { echo "Missing configured DICOM directory: ${DICOM_DIR}" >&2; exit 1; }
  mkdir -p "$(dirname "${MAT_PATH}")"
  matlab -batch "addpath('${CODE10W_ROOT}/quality_experiments/Q001_full_fov/preprocessing'); addpath('${CODE10W_ROOT}/trad/modules/module_01_preprocess_matlab/preprocessing_v1'); opts=preprocess_options_v1([],false,true); preprocess_stack_full_fov_q001('${DICOM_DIR}','${MAT_PATH}',opts);"
  conda run --no-capture-output -n knesvr_torch python -m quality_experiments.Q001_full_fov.bridge.full_fov_bridge --preprocessed-mat "${MAT_PATH}" --dicom-dir "${DICOM_DIR}" --stack "${stack}" --output-dir "${PREPARED_DIR}"
done
