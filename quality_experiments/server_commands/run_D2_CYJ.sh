#!/usr/bin/env bash
set -euo pipefail

CODE10W_ROOT=/data/dengyz/code/Code_10w
DATA_ROOT=/data/dengyz/dataset/Code_10w_v1
SUBJECT_ID=CYJ
BASELINE_EXP_ROOT="${DATA_ROOT}/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1"
QUALITY_ROOT="${DATA_ROOT}/Code_10w_runs/quality_experiments_v1"
TRAD_RUN="${BASELINE_EXP_ROOT}/runs/trad_B6"
MLP_RUN="${BASELINE_EXP_ROOT}/runs/frozen_mlp_B6"
PREPARED_ROOT="${BASELINE_EXP_ROOT}/inputs/prepared"
OUTPUT_ROOT="${QUALITY_ROOT}/${SUBJECT_ID}/D2_k8_signal_domain_baseline_v1"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cr_dreme
cd "${CODE10W_ROOT}"
COMMIT_SHA="$(git rev-parse HEAD)"
python -c 'import torch; assert torch.cuda.is_available(), "D2 formal K=8 export requires CUDA on cr_dreme"'

for required in "${TRAD_RUN}/model.pt" "${MLP_RUN}/model.pt" "${PREPARED_ROOT}"; do
  [[ -e "${required}" ]] || { echo "Missing required input: ${required}" >&2; exit 1; }
done
if [[ -f "${PREPARED_ROOT}/${SUBJECT_ID}/sax/observations.npz" && -f "${PREPARED_ROOT}/${SUBJECT_ID}/2ch/observations.npz" && -f "${PREPARED_ROOT}/${SUBJECT_ID}/4ch/observations.npz" ]]; then
  : # nested subject layout
elif [[ -f "${PREPARED_ROOT}/sax/observations.npz" && -f "${PREPARED_ROOT}/2ch/observations.npz" && -f "${PREPARED_ROOT}/4ch/observations.npz" ]]; then
  : # flat baseline-provenance layout
else
  echo "Missing complete prepared observations in either explicit D2 layout below ${PREPARED_ROOT}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}" ]] && [[ -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty D2 output root: ${OUTPUT_ROOT}" >&2
  exit 1
fi

RUN_LOG="$(mktemp -t code10w-d2.XXXXXX.log)"
printf '%s\n' "D2 implementation: formal K=8 signal-domain inference" "Git commit: ${COMMIT_SHA}" | tee "${RUN_LOG}"
python quality_experiments/D2_k8_signal_domain/run_d2_k8_signal_domain.py \
  --subject-id "${SUBJECT_ID}" \
  --trad-run "${TRAD_RUN}" \
  --mlp-run "${MLP_RUN}" \
  --prepared-root "${PREPARED_ROOT}" \
  --protocol trad/configs/protocol_hhz_v1.yaml \
  --output "${OUTPUT_ROOT}" \
  --device cuda:0 2>&1 | tee -a "${RUN_LOG}"
cp "${RUN_LOG}" "${OUTPUT_ROOT}/logs/run.log"
printf '%s\n' "D2 outputs: ${OUTPUT_ROOT}" "Summary: ${OUTPUT_ROOT}/RESULT_SUMMARY.md"
