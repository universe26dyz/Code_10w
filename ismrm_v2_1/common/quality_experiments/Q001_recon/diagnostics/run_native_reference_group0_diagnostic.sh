#!/usr/bin/env bash
set -u -o pipefail

CODE10W_ROOT=/home/universe/SVR/multimap_postprogramming/Code_10w
INPUT_MAT=/home/universe/SVR/data/Code_10w_v1/Q001_full_fov_inputs_v2/full_fov_preprocessed/CYJ/sax/preprocessed.mat
DIAGNOSTIC_ROOT="${Q001_DIAGNOSTIC_ROOT:-/home/universe/SVR/data/Code_10w_v1/Q001_full_fov_reference_diagnostic_v2}"
POOL1=false
if [[ "${1:-}" == "--pool1" ]]; then POOL1=true; DIAGNOSTIC_ROOT="${Q001_DIAGNOSTIC_ROOT:-/home/universe/SVR/data/Code_10w_v1/Q001_full_fov_reference_diagnostic_v2_pool1}"; fi
[[ -f "${INPUT_MAT}" ]] || { echo "Missing input MAT: ${INPUT_MAT}" >&2; exit 2; }
[[ ! -e "${DIAGNOSTIC_ROOT}" || -z "$(find "${DIAGNOSTIC_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || { echo "Refusing non-empty diagnostic root: ${DIAGNOSTIC_ROOT}" >&2; exit 2; }
mkdir -p "${DIAGNOSTIC_ROOT}"
START_EPOCH="$(date +%s)"; START_ISO="$(date --iso-8601=seconds)"; HOST="$(hostname)"
LOG="${DIAGNOSTIC_ROOT}/matlab.log"; OUTPUT="${DIAGNOSTIC_ROOT}/sax_group0.mat"
GRID_JSON="${DIAGNOSTIC_ROOT}/dictionary_grid.json"
free -h > "${DIAGNOSTIC_ROOT}/memory_before.txt" 2>&1 || true
MATLAB_VERSION="$(matlab -batch "disp(version)" 2>&1 || true)"
PCT="$(matlab -batch "disp(license('test','Distrib_Computing_Toolbox'))" 2>&1 || true)"
matlab -batch "addpath('${CODE10W_ROOT}/quality_experiments/Q001_recon/matlab'); q001_dictionary_grid_report('${GRID_JSON}');" >/dev/null
PREFIX=""
if [[ "${POOL1}" == true ]]; then PREFIX="parpool('local',1);"; fi
set +e
matlab -batch "${PREFIX} addpath('${CODE10W_ROOT}/quality_experiments/Q001_recon/matlab'); q001_one_group_reference_diagnostic('${INPUT_MAT}','${OUTPUT}');" -logfile "${LOG}"
MATLAB_EXIT=$?
set -e
END_EPOCH="$(date +%s)"; END_ISO="$(date --iso-8601=seconds)"; WALL=$((END_EPOCH-START_EPOCH))
free -h > "${DIAGNOSTIC_ROOT}/memory_after.txt" 2>&1 || true
find "${HOME}" /tmp -maxdepth 2 -type f \( -name 'matlab_crash_dump.*' -o -name 'java.log.*' \) -newermt "${START_ISO}" -printf '%p\n' 2>/dev/null | sort > "${DIAGNOSTIC_ROOT}/crash_dump_files.txt" || true
search_runtime_logs() {
  if command -v rg >/dev/null 2>&1; then
    rg -i 'killed process|out of memory|oom-killer|matlab'
  else
    grep -Ei 'killed process|out of memory|oom-killer|matlab'
  fi
}
(dmesg --time-format iso 2>&1 | tail -n 400 | search_runtime_logs || true) > "${DIAGNOSTIC_ROOT}/oom_evidence.txt"
(journalctl --since "${START_ISO}" 2>&1 | search_runtime_logs || true) >> "${DIAGNOSTIC_ROOT}/oom_evidence.txt"
OUTPUT_SIZE=0; [[ -f "${OUTPUT}" ]] && OUTPUT_SIZE="$(stat -c %s "${OUTPUT}")"
export START_ISO END_ISO HOST LOG OUTPUT MATLAB_EXIT WALL OUTPUT_SIZE POOL1 MATLAB_VERSION PCT DIAGNOSTIC_ROOT
python3 -c "import json,os; r=os.environ['DIAGNOSTIC_ROOT']; p=lambda n:open(os.path.join(r,n),encoding='utf-8',errors='replace').read().splitlines(); grid=json.load(open(os.path.join(r,'dictionary_grid.json'))); x={'start_time':os.environ['START_ISO'],'end_time':os.environ['END_ISO'],'hostname':os.environ['HOST'],'matlab_exit_code':int(os.environ['MATLAB_EXIT']),'wall_time_sec':int(os.environ['WALL']),'output_exists':os.path.isfile(os.environ['OUTPUT']),'output_size_bytes':int(os.environ['OUTPUT_SIZE']),'log_path':os.environ['LOG'],'matlab_version':os.environ['MATLAB_VERSION'],'parallel_toolbox_available':os.environ['PCT'],'parallel_pool_state':'pool1_requested' if os.environ['POOL1']=='true' else 'default_matcher_parallel_behavior','crash_dump_files':p('crash_dump_files.txt'),'oom_evidence':p('oom_evidence.txt'),'dictionary_grid':grid,'status':'SUCCESS' if os.path.isfile(os.environ['OUTPUT']) and int(os.environ['MATLAB_EXIT'])==0 else 'PROCESS_DIAGNOSTIC_CAPTURED'}; open(os.path.join(r,'diagnostic_summary.json'),'w').write(json.dumps(x,indent=2)+'\\n')"
exit "${MATLAB_EXIT}"
