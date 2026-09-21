#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# Code_10w overnight Wave-1 automation
#   Trad: B6 -> TV2 -> TV3, with server-side evaluation after each run
#   MLP : RR pilot generation -> training -> synthetic validation
#         -> real VPS87 validation -> decoder GPU benchmark
#
# Safe behavior:
#   - reconstruction / dataset generation / MLP training failures STOP the chain
#   - evaluation-only failures are logged and the next experiment continues
#   - existing completed outputs are reused; partial non-empty outputs abort
#   - never deletes or overwrites an existing scientific result
# =============================================================================

EXPECTED_HEAD="${EXPECTED_HEAD:-122034a56584e364f218ff4c68d6323ef924fa8a}"
REPO="${REPO:-/data/dengyz/code/Code_10w}"
DATA="${DATA:-/data/dengyz/dataset/Code_10w_v1}"
SUBJECT="${SUBJECT:-CYJ}"
ENV_NAME="${ENV_NAME:-cr_dreme}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-20260921}"

PREPARED="${PREPARED:-${DATA}/Code_10w_prepared}"
PREPROCESSED="${PREPROCESSED:-${DATA}/Code_10w_preprocessed}"
TRAD_RUN_ROOT="${TRAD_RUN_ROOT:-${DATA}/Code_10w_runs/trad_v3}"
MLP_ROOT="${MLP_ROOT:-${DATA}/mlp_rr_v3}"
DICT_CACHE="${DICT_CACHE:-${DATA}/dictionary_cache/${SUBJECT}}"
OVERNIGHT_ROOT="${OVERNIGHT_ROOT:-${DATA}/Code_10w_runs/overnight_wave1_20260921}"
LOG_ROOT="${OVERNIGHT_ROOT}/logs"
STATUS_ROOT="${OVERNIGHT_ROOT}/status"

mkdir -p "${LOG_ROOT}" "${STATUS_ROOT}" "${TRAD_RUN_ROOT}" "${MLP_ROOT}" "${DICT_CACHE}"

MASTER_LOG="${OVERNIGHT_ROOT}/master.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }
die() { log "FATAL: $*"; exit 1; }

on_err() {
  local rc=$?
  log "FATAL: command failed at line ${BASH_LINENO[0]} with exit code ${rc}"
  exit "${rc}"
}
trap on_err ERR

is_nonempty_dir() {
  [[ -d "$1" ]] && [[ -n "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]
}

first_existing_dir() {
  local p
  for p in "$@"; do
    if [[ -d "${p}" ]]; then
      printf '%s\n' "${p}"
      return 0
    fi
  done
  return 1
}

optional_eval() {
  local label="$1"
  shift
  local logfile="${LOG_ROOT}/${label}.log"
  log "EVAL START: ${label}"
  set +e
  "$@" 2>&1 | tee "${logfile}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ ${rc} -eq 0 ]]; then
    touch "${STATUS_ROOT}/${label}.PASS"
    log "EVAL PASS: ${label}"
  else
    echo "${rc}" > "${STATUS_ROOT}/${label}.FAIL"
    log "EVAL FAIL (non-fatal, continuing): ${label}, rc=${rc}"
  fi
  return 0
}

optional_eval_shell() {
  local label="$1"
  local command="$2"
  optional_eval "${label}" bash -lc "${command}"
}

# -----------------------------------------------------------------------------
# Preflight
# -----------------------------------------------------------------------------
log "===== PRE-FLIGHT ====="
[[ -d "${REPO}/.git" ]] || die "Repository missing: ${REPO}"

cd "${REPO}"
HEAD="$(git rev-parse HEAD)"
log "Git HEAD: ${HEAD}"
[[ "${HEAD}" == "${EXPECTED_HEAD}" ]] || die "Expected HEAD ${EXPECTED_HEAD}, got ${HEAD}. Pull/verify source before overnight run."

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  die "Tracked repository files are modified. Refusing an unattended scientific run."
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found."
nvidia-smi -L
conda run --no-capture-output -n "${ENV_NAME}" python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print("gpu", torch.cuda.get_device_name(0))
PY

for stack in sax 2ch 4ch; do
  [[ -f "${PREPARED}/${SUBJECT}/${stack}/observations.npz" ]] || die "Missing prepared observation: ${PREPARED}/${SUBJECT}/${stack}/observations.npz"
done

# Reusable evaluation assets are optional on the server. If absent, the script
# still runs K=8 GPU Stage-1 and map-domain K32 reprojection without reference metrics.
NATIVE_REFERENCE="${NATIVE_REFERENCE:-}"
MASK_BUNDLE="${MASK_BUNDLE:-}"

if [[ -z "${NATIVE_REFERENCE}" ]]; then
  NATIVE_REFERENCE="$(first_existing_dir \
    "${DATA}/evaluation_assets/${SUBJECT}/native_reference_2d" \
    "${DATA}/Code_10w_runs/trad_v2/${SUBJECT}_stackinit_TV_baseline/native_reference_2d" \
    2>/dev/null || true)"
fi

if [[ -z "${MASK_BUNDLE}" ]]; then
  MASK_BUNDLE="$(first_existing_dir \
    "${DATA}/evaluation_assets/${SUBJECT}/myocardium_masks_v2" \
    "${DATA}/Code_10w_runs/trad_v2/${SUBJECT}_stackinit_TV_baseline/evaluation_scmr/myocardium_masks_v2" \
    2>/dev/null || true)"
fi

FULL_REFERENCE_EVAL=0
if [[ -n "${NATIVE_REFERENCE}" && -f "${NATIVE_REFERENCE}/native_reference_manifest.json" \
   && -n "${MASK_BUNDLE}" && -f "${MASK_BUNDLE}/manifest.json" ]]; then
  FULL_REFERENCE_EVAL=1
  log "Reusable native reference: ${NATIVE_REFERENCE}"
  log "Reusable myocardium masks: ${MASK_BUNDLE}"
else
  log "NOTE: native-reference/mask bundle not both available on server."
  log "      K32 map-domain reprojection will still run, but reference/ROI metrics will be deferred."
fi

MATLAB_AVAILABLE=0
if command -v matlab >/dev/null 2>&1; then
  MATLAB_AVAILABLE=1
  log "MATLAB detected: $(command -v matlab)"
else
  log "NOTE: MATLAB not found on server; K=8 Stage-2 apparent-map postprocess will be deferred to local workstation."
fi

# -----------------------------------------------------------------------------
# One Trad experiment + evaluations
# -----------------------------------------------------------------------------
run_trad() {
  local exp="$1"
  local cfg="${REPO}/trad/configs/experiments_6k/${exp}.yaml"
  local run="${TRAD_RUN_ROOT}/${SUBJECT}_${exp}"
  local train_log="${LOG_ROOT}/${exp}_train.log"

  [[ -f "${cfg}" ]] || die "Missing config: ${cfg}"

  log "======================================================================"
  log "TRAD ${exp}: START"
  log "Output: ${run}"

  if [[ -f "${run}/model.pt" && -f "${run}/T1_3D.nii.gz" && -f "${run}/T2_3D.nii.gz" \
     && -f "${run}/final_rigid_poses.json" && -f "${run}/config_resolved.yaml" ]]; then
    log "TRAD ${exp}: completed output already exists; training skipped."
  elif is_nonempty_dir "${run}"; then
    die "TRAD ${exp}: non-empty partial output exists at ${run}; refusing overwrite/resume ambiguity."
  else
    (
      cd "${REPO}/trad"
      conda run --no-capture-output -n "${ENV_NAME}" \
        python -m scripts.reconstruct_subject \
          --prepared-root "${PREPARED}" \
          --subject-id "${SUBJECT}" \
          --config "configs/experiments_6k/${exp}.yaml" \
          --output "${run}"
    ) 2>&1 | tee "${train_log}"
    touch "${STATUS_ROOT}/${exp}_TRAIN.PASS"
  fi

  [[ -f "${run}/model.pt" ]] || die "${exp}: model.pt missing after training."
  [[ -f "${run}/timing_profile.csv" ]] || die "${exp}: timing_profile.csv missing after training."
  [[ -f "${run}/config_resolved.yaml" ]] || die "${exp}: config_resolved.yaml missing after training."
  [[ -f "${run}/T1_3D.nii.gz" && -f "${run}/T2_3D.nii.gz" ]] || die "${exp}: exported quantitative maps missing."
  [[ -f "${run}/final_rigid_poses.json" ]] || die "${exp}: final_rigid_poses.json missing."

  # 1) Profiler summary
  optional_eval_shell "${exp}_timing_summary" \
    "cd '${REPO}' && conda run --no-capture-output -n '${ENV_NAME}' python -m trad.evaluation.profiling.summarize_timing_profile \
      --timing-profile '${run}/timing_profile.csv' \
      --config '${run}/config_resolved.yaml'"

  # 2) Physics-consistent signal-domain K=8 Stage-1 on GPU
  local stage1="${run}/evaluation_scmr/psf_k8_reprojection_stage1"
  if [[ -f "${stage1}/stage1_manifest.json" ]]; then
    log "${exp}: K8 Stage-1 already complete; skipped."
  elif is_nonempty_dir "${stage1}"; then
    log "${exp}: WARNING non-empty incomplete K8 Stage-1 directory; not overwriting."
    echo "partial output" > "${STATUS_ROOT}/${exp}_psf_k8_stage1.FAIL"
  else
    optional_eval_shell "${exp}_psf_k8_stage1" \
      "cd '${REPO}' && conda run --no-capture-output -n '${ENV_NAME}' python -m trad.evaluation.scmr.run_psf_k8_gpu_stage \
        --subject-id '${SUBJECT}' \
        --prepared-root '${PREPARED}' \
        --checkpoint '${run}/model.pt' \
        --config '${run}/config_resolved.yaml' \
        --final-poses '${run}/final_rigid_poses.json' \
        --output '${stage1}' \
        --device '${DEVICE}'"
  fi

  # 3) Common exported-volume map-domain PSF K=32 screening
  local map_out
  local ref_args=""
  if [[ "${FULL_REFERENCE_EVAL}" -eq 1 ]]; then
    map_out="${run}/evaluation_scmr/map_domain_psf_k32"
    ref_args="--native-reference '${NATIVE_REFERENCE}' --preprocessed-root '${PREPROCESSED}' --mask-bundle '${MASK_BUNDLE}'"
  else
    map_out="${run}/evaluation_scmr/map_domain_psf_k32_reprojection_only"
  fi

  if [[ -f "${map_out}/map_domain_psf_manifest.json" ]]; then
    log "${exp}: map-domain K32 already complete; skipped."
  elif is_nonempty_dir "${map_out}"; then
    log "${exp}: WARNING non-empty incomplete map-domain K32 directory; not overwriting."
    echo "partial output" > "${STATUS_ROOT}/${exp}_map_domain_k32.FAIL"
  else
    optional_eval_shell "${exp}_map_domain_k32" \
      "cd '${REPO}' && conda run --no-capture-output -n '${ENV_NAME}' python -m trad.evaluation.scmr.run_map_domain_psf_comparison \
        --method trad \
        --subject-id '${SUBJECT}' \
        --t1-volume '${run}/T1_3D.nii.gz' \
        --t2-volume '${run}/T2_3D.nii.gz' \
        --prepared-root '${PREPARED}' \
        --final-poses '${run}/final_rigid_poses.json' \
        --n-samples 32 \
        --seed '${SEED}' \
        ${ref_args} \
        --output '${map_out}'"
  fi

  # 4) Optional full K=8 Stage-2 apparent-map matching/metrics.
  #    This is run only if MATLAB + reusable native reference/masks are present.
  if [[ "${MATLAB_AVAILABLE}" -eq 1 && "${FULL_REFERENCE_EVAL}" -eq 1 && -f "${stage1}/stage1_manifest.json" ]]; then
    local stage2="${run}/evaluation_scmr/psf_k8_evaluation_v2"
    if [[ -f "${stage2}/evaluation_manifest.json" ]]; then
      log "${exp}: K8 Stage-2 already complete; skipped."
    elif is_nonempty_dir "${stage2}"; then
      log "${exp}: WARNING non-empty incomplete K8 Stage-2 directory; not overwriting."
      echo "partial output" > "${STATUS_ROOT}/${exp}_psf_k8_stage2.FAIL"
    else
      optional_eval_shell "${exp}_psf_k8_stage2" \
        "cd '${REPO}' && conda run --no-capture-output -n '${ENV_NAME}' python -m trad.evaluation.scmr.run_psf_k8_postprocess \
          --subject-id '${SUBJECT}' \
          --stage1-root '${stage1}' \
          --preprocessed-root '${PREPROCESSED}' \
          --native-reference '${NATIVE_REFERENCE}' \
          --mask-bundle '${MASK_BUNDLE}' \
          --dictionary-cache-root '${DICT_CACHE}' \
          --output '${stage2}' \
          --matlab-executable matlab"
    fi
  else
    log "${exp}: K8 Stage-2 deferred (requires MATLAB + native reference + myocardium mask bundle)."
  fi

  log "TRAD ${exp}: FINISHED"
  sync
}

# -----------------------------------------------------------------------------
# Trad Wave-1: sequential, same GPU, one finishes before the next starts
# -----------------------------------------------------------------------------
log "===== TRAD WAVE-1 START ====="
run_trad B6
run_trad TV2
run_trad TV3
log "===== TRAD WAVE-1 FINISHED ====="

# -----------------------------------------------------------------------------
# RR-MLP pilot: generation -> training -> validations -> decoder benchmark
# -----------------------------------------------------------------------------
log "===== RR-MLP PILOT START ====="
PILOT_DATA="${MLP_ROOT}/pilot"
PILOT_MODEL="${MLP_ROOT}/pilot_model"
PILOT_CFG="${REPO}/mlp/configs/rr_synthetic/pilot_signalonly.yaml"
PROTOCOL="${REPO}/mlp/configs/protocol_hhz_v1.yaml"

# 1) 250k synthetic RR pilot generation
if [[ -f "${PILOT_DATA}/dataset_metadata.json" && -f "${PILOT_DATA}/train.h5" \
   && -f "${PILOT_DATA}/valid.h5" && -f "${PILOT_DATA}/test.h5" ]]; then
  log "RR pilot dataset already complete; generation skipped."
elif is_nonempty_dir "${PILOT_DATA}"; then
  die "RR pilot dataset directory is non-empty but incomplete: ${PILOT_DATA}"
else
  (
    cd "${REPO}/mlp"
    conda run --no-capture-output -n "${ENV_NAME}" \
      python -m modules.module_05_signal_decoder.generate_rr_mlp_dataset \
        --config "configs/rr_synthetic/pilot_signalonly.yaml" \
        --output-dir "${PILOT_DATA}"
  ) 2>&1 | tee "${LOG_ROOT}/MLP_pilot_generation.log"
  touch "${STATUS_ROOT}/MLP_PILOT_GENERATION.PASS"
fi

# 2) 50-epoch pilot MLP training
if [[ -f "${PILOT_MODEL}/signal_simulator_best.pth" && -f "${PILOT_MODEL}/test_metrics.json" ]]; then
  log "RR pilot model already complete; training skipped."
elif is_nonempty_dir "${PILOT_MODEL}"; then
  die "RR pilot model directory is non-empty but incomplete: ${PILOT_MODEL}"
else
  (
    cd "${REPO}/mlp"
    conda run --no-capture-output -n "${ENV_NAME}" \
      python -m modules.module_05_signal_decoder.train_mlp \
        --config "configs/rr_synthetic/pilot_signalonly.yaml" \
        --dataset-dir "${PILOT_DATA}" \
        --protocol "configs/protocol_hhz_v1.yaml" \
        --output-dir "${PILOT_MODEL}"
  ) 2>&1 | tee "${LOG_ROOT}/MLP_pilot_train.log"
  touch "${STATUS_ROOT}/MLP_PILOT_TRAIN.PASS"
fi

CHECKPOINT="${PILOT_MODEL}/signal_simulator_best.pth"
[[ -f "${CHECKPOINT}" ]] || die "Pilot checkpoint missing: ${CHECKPOINT}"

# 3) Explicit synthetic held-out validation
if [[ -f "${PILOT_MODEL}/synthetic_validation.json" ]]; then
  log "Synthetic validation already exists; skipped."
else
  optional_eval_shell "MLP_pilot_synthetic_validation" \
    "cd '${REPO}/mlp' && conda run --no-capture-output -n '${ENV_NAME}' python -m modules.module_05_signal_decoder.validate_formal_mlp \
      --checkpoint '${CHECKPOINT}' \
      --dataset-dir '${PILOT_DATA}' \
      --protocol 'configs/protocol_hhz_v1.yaml' \
      --output '${PILOT_MODEL}/synthetic_validation.json' \
      --device '${DEVICE}' \
      --batch-size 1024 \
      --gradient-samples 64"
fi

# 4) Real CYJ/DYZ/HHZ/HJL VPS87 timing validation
REAL_VALIDATION="${PILOT_MODEL}/real_timing_validation.json"
if [[ -f "${REAL_VALIDATION}" ]]; then
  log "Real VPS87 timing validation already exists; skipped."
else
  optional_eval_shell "MLP_pilot_real_vps87_validation" \
    "cd '${REPO}/mlp' && conda run --no-capture-output -n '${ENV_NAME}' python -m modules.module_05_signal_decoder.validate_real_timings \
      --checkpoint '${CHECKPOINT}' \
      --prepared-root '${PREPARED}' \
      --subjects CYJ DYZ HHZ HJL \
      --rr-dataset-dir '${PILOT_DATA}' \
      --protocol 'configs/protocol_hhz_v1.yaml' \
      --output '${REAL_VALIDATION}' \
      --device '${DEVICE}' \
      --samples-per-timing 16"
fi

# 5) Decoder-only Trad-vs-MLP speed/fidelity benchmark
BENCHMARK="${PILOT_MODEL}/decoder_rr_benchmark.json"
if [[ -f "${BENCHMARK}" ]]; then
  log "RR decoder benchmark already exists; skipped."
else
  optional_eval_shell "MLP_pilot_decoder_benchmark" \
    "cd '${REPO}/mlp' && conda run --no-capture-output -n '${ENV_NAME}' python -m modules.module_09_qc_benchmark.benchmark_signal_decoder \
      --checkpoint '${CHECKPOINT}' \
      --rr-dataset-dir '${PILOT_DATA}' \
      --protocol 'configs/protocol_hhz_v1.yaml' \
      --output '${BENCHMARK}' \
      --device '${DEVICE}' \
      --batch-size 2560 \
      --batch-size 5120 \
      --batch-size 10240 \
      --repetitions 3"
fi

log "===== RR-MLP PILOT FINISHED ====="

# -----------------------------------------------------------------------------
# Compact machine-readable / human-readable overnight summary
# -----------------------------------------------------------------------------
log "===== BUILD OVERNIGHT SUMMARY ====="
conda run -n "${ENV_NAME}" python - "${TRAD_RUN_ROOT}" "${MLP_ROOT}" "${OVERNIGHT_ROOT}" "${SUBJECT}" <<'PY'
import csv, json, math, sys
from pathlib import Path

trad_root = Path(sys.argv[1])
mlp_root = Path(sys.argv[2])
out_root = Path(sys.argv[3])
subject = sys.argv[4]
summary = {"trad": {}, "mlp": {}}
rows = []

def safe_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None

for exp in ("B6", "TV2", "TV3"):
    run = trad_root / f"{subject}_{exp}"
    rec = {
        "run": str(run),
        "train_complete": (run / "model.pt").is_file(),
        "timing_summary": None,
        "map_domain_k32": None,
        "physics_k8_stage1": (run / "evaluation_scmr/psf_k8_reprojection_stage1/stage1_manifest.json").is_file(),
        "physics_k8_stage2": (run / "evaluation_scmr/psf_k8_evaluation_v2/evaluation_manifest.json").is_file(),
    }
    timing = safe_json(run / "timing_profile_summary.json")
    if timing:
        rec["timing_summary"] = {
            "estimated_total_training_seconds": timing.get("estimated_total_training_seconds")
        }
    map_metrics = safe_json(run / "evaluation_scmr/map_domain_psf_k32/map_domain_psf_metrics.json")
    if map_metrics:
        rec["map_domain_k32"] = map_metrics
    stage2 = safe_json(run / "evaluation_scmr/psf_k8_evaluation_v2/metrics/metrics_summary.json")
    if stage2:
        rec["physics_k8_stage2_metrics"] = stage2
    summary["trad"][exp] = rec

    row = {"experiment": exp, "train_complete": rec["train_complete"],
           "training_est_s": (rec["timing_summary"] or {}).get("estimated_total_training_seconds", ""),
           "k8_stage1": rec["physics_k8_stage1"], "k8_stage2": rec["physics_k8_stage2"]}
    gm = (map_metrics or {}).get("global_common_support", {})
    for p in ("T1", "T2"):
        m = gm.get(p, {})
        row[f"map_{p}_RMSE_ms"] = m.get("RMSE_ms", "")
        row[f"map_{p}_MAE_ms"] = m.get("MAE_ms", "")
        row[f"map_{p}_bias_ms"] = m.get("bias_ms", "")
        row[f"map_{p}_NCC"] = m.get("NCC", "")
    rows.append(row)

pilot_model = mlp_root / "pilot_model"
summary["mlp"]["pilot_test_metrics"] = safe_json(pilot_model / "test_metrics.json")
summary["mlp"]["synthetic_validation"] = safe_json(pilot_model / "synthetic_validation.json")
summary["mlp"]["real_timing_validation"] = safe_json(pilot_model / "real_timing_validation.json")
summary["mlp"]["decoder_benchmark"] = safe_json(pilot_model / "decoder_rr_benchmark.json")

out_root.mkdir(parents=True, exist_ok=True)
(out_root / "overnight_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
with (out_root / "trad_wave1_summary.csv").open("w", newline="") as f:
    fields = ["experiment","train_complete","training_est_s","k8_stage1","k8_stage2",
              "map_T1_RMSE_ms","map_T1_MAE_ms","map_T1_bias_ms","map_T1_NCC",
              "map_T2_RMSE_ms","map_T2_MAE_ms","map_T2_bias_ms","map_T2_NCC"]
    w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

print("summary:", out_root / "overnight_summary.json")
print("trad csv:", out_root / "trad_wave1_summary.csv")
PY

touch "${STATUS_ROOT}/ALL_SCHEDULED_TASKS_FINISHED"
log "======================================================================"
log "ALL SCHEDULED TASKS FINISHED"
log "Master log: ${MASTER_LOG}"
log "Summary: ${OVERNIGHT_ROOT}/overnight_summary.json"
log "Trad table: ${OVERNIGHT_ROOT}/trad_wave1_summary.csv"
log "Status markers: ${STATUS_ROOT}"
log "======================================================================"
