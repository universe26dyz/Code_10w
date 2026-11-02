"""Generate bilingual operation manuals after a reproducibility archive is complete."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _zh(subject: str, manifest: dict[str, Any], execution_commit: str) -> str:
    return f"""# SCMR-v5 MultiMap 可复现实验归档（中文）

本目录是已经完成物理复制与 SHA256 登记的可复现实验归档。它保存一个实际 subject 示例（{subject}），但软件入口均使用显式 SUBJECT 参数，适用于其他 subject。归档本身是只读证据；所有重新运行都必须写到新的工作目录，绝不能把新结果写回这里。

## 目录用途

~~~
00_live_route_freeze/                 原始分散成功路线的冻结证据
01_code_snapshot/source.tar           可恢复的代码快照
01_code_snapshot/source.bundle        Git 历史 bundle
01_execution_code/code/               可直接复制使用的、未压缩项目代码树
01_execution_code/00_pipeline_code_map/ 按运行步骤索引源码
02_runtime_configuration/             独立、已哈希的通用配置与示例 provenance
03_subject_examples/{subject}/01_inputs/
  raw_dicom/ deployment_manifest.json preprocessed/ prepared/ roi/ sparse_inputs/
03_subject_examples/{subject}/02_outputs/
  mapping/ native_cache/ fit_first/ proposed/ final_pose/ common_native/
99_integrity/                         archive_manifest 与逐文件 SHA256
~~~

冻结证据代码 commit 为 {manifest["code_snapshot"]["commit"]}；实际运行的可见代码树 commit 为 {execution_commit}。共登记 {manifest["file_count"]} 个复制文件。

## 从原始 DICOM 开始：通用运行顺序

### 0. 先取出代码，并建立本次运行的通用配置副本

`01_code_snapshot/` 是本次成功路线的只读证据；实际操作使用 `01_execution_code/code/` 这一未压缩代码树。先将它复制到新的工作目录；之后的改动不能发生在归档目录内。`02_runtime_configuration/00_common/` 保存不带 subject 名称的正确 protocol 和 Proposed 配置；其逐文件 SHA 位于同目录的 `runtime_configuration_manifest.jsonl`。

~~~bash
ARCHIVE="/path/to/01_reproducible_pipeline"
WORK_CODE="/path/to/new_work/Code_10w"
mkdir -p "$WORK_CODE"
cp -a "$ARCHIVE/01_execution_code/code/." "$WORK_CODE/"
cp "$ARCHIVE/02_runtime_configuration/00_common/protocol_scmr_v5_10_100.yaml" \
   "$WORK_CODE/mlp/configs/protocol_scmr_v5_10_100.yaml"
PROTOCOL="$WORK_CODE/mlp/configs/protocol_scmr_v5_10_100.yaml"
sha256sum "$PROTOCOL"
~~~

### 运行根目录结构（服务器）

每个新 subject 使用一个新的、空的 `$RUN_ROOT`；所有自动生成文件均在其中。目录名与现有 generic launcher 完全兼容：

~~~text
$RUN_ROOT/
  config/                 单 subject deployment manifest
  preprocessed/<SUBJECT>/ 服务器 MATLAB 预处理 MAT 与 QC
  prepared/<SUBJECT>/     observations.npz、timing、bridge manifest/QC
  sparse_inputs/<SUBJECT>/稀疏采样 profile（Proposed 与 fit-first 共用）
  mapping/scmr_v5/        MATLAB fitting map
  native_cache/           SHA 绑定的 native T1/T2/B1 cache
  reconstruction/
    fit_first/<SUBJECT>/  2D-fit-first 输出
    proposed/<SUBJECT>/   Proposed 输出
  roi_manual/             本地人工勾画后上传的 ROI bundle
  evaluation/final_pose/  final pose、AHA、指标与 QC 图
  01_server_preprocess.log
  02_preprocessed_validation.log
  03_prepare_observations.log
  server_matlab_version.txt
~~~

### 1. 服务器：原始 DICOM 与单 subject manifest

新 subject 的正式计算全部在服务器完成。服务器必须保存 sax、2ch、4ch 三个原始 DICOM 目录；每个 slice/group 恰有十个 weight。NIfTI 不能替代 DICOM，因为后续需要 SOP UID、采集时间和 DICOM 空间几何。若 DICOM 当前仅在本地，本地只做一次原始数据上传；之后不再在本地运行 MATLAB 或 Python。

将 `configs/subject_stack_manifest.example.json` 复制到代码目录外。它仍保留历史字段名 `local_dicom_dir`，但 server-only 运行时，`local_dicom_dir` 与 `server_dicom_dir` 必须都填写为**同一服务器 DICOM 路径**；每次 manifest 只含一个 subject。

~~~bash
SUBJECT="NEW_SUBJECT"
PY="/path/to/server/python"
RUN_ROOT="/path/to/new_server_run/$SUBJECT"
MANIFEST="$RUN_ROOT/config/$SUBJECT-deployment.json"
SERVER_DICOM_ROOT="/path/to/server_dicom"
mkdir -p "$RUN_ROOT/config"
cp "$WORK_CODE/configs/subject_stack_manifest.example.json" "$MANIFEST"

for STACK in sax 2ch 4ch; do
  N=$(find "$SERVER_DICOM_ROOT/$SUBJECT/$STACK" -maxdepth 1 -name '*.dcm' | wc -l)
  echo "$STACK: $N DICOM files"
  test "$N" -gt 0 && test $((N % 10)) -eq 0
done
~~~

编辑 `$MANIFEST`：三个 stack 的两个 DICOM 字段均填为 `$SERVER_DICOM_ROOT/$SUBJECT/<stack>`。人工勾画 ROI 是唯一保留在本地的工作：它在重建完成后生成 ROI bundle（至少包含 `sax_myocardium_masks.npz` 和 `roi_derivation_manifest.json`），然后只上传该 bundle 到 `$RUN_ROOT/roi_manual/`，供评估使用。

### 2. 服务器：DICOM → `preprocessed.mat` → `observations.npz`

先记录服务器 MATLAB 环境；随后 launcher 在服务器上执行 MATLAB 的 crop、MIND、MP-PCA，并执行 Python bridge。`local_preprocess_all.m` 是历史函数名，但由该 launcher 在服务器运行；不再传输 MAT。输出为 `$RUN_ROOT/preprocessed/`、`$RUN_ROOT/prepared/` 和三个日志文件。

~~~bash
SERVER_MATLAB_REPORT="$RUN_ROOT/server_matlab_version.txt"
matlab -batch "disp(version); ver" > "$SERVER_MATLAB_REPORT" 2>&1

bash "$WORK_CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/server_preprocess_and_prepare.sh" \
  "$RUN_ROOT" "$SUBJECT" "$MANIFEST" "$PROTOCOL" matlab "$PY"
~~~

通过条件：三个 stack 均有 `preprocessed.mat`、`preprocess_qc.json`、`observations.npz`、`manifest.json`；MAT 的 `Mag_crop` 第三维为 10，QC 为 MP-PCA enabled、`center_data=true`。MAT 保留真实 acquisition/preprocessing metadata 的 `TI=[50,150]`；SCMR-v5 matching/observation timing 通过 `$PROTOCOL` 使用 `TI=[10,100]`，TR/VPS 仍来自 DICOM/MAT。

### 3. 服务器：生成稀疏 case 输入

该 Python 步骤从服务器的完整 `observations.npz` 派生所有标准 case，并记录每个 retained/held-out group 与 SHA。它不更改 full observations。输出写入 `$RUN_ROOT/sparse_inputs/$SUBJECT/`，供 Proposed 与 2D-fit-first 共同使用。

~~~bash
bash "$WORK_CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/build_sparse_profiles.sh" \
  "$RUN_ROOT/prepared" "$SUBJECT" "$RUN_ROOT/sparse_inputs" "$PY"
~~~

### 4. 服务器：SCMR-v5 fitting 与 2D-fit-first

~~~
FIT_RUN="$RUN_ROOT"
PY="$PY" NESVOR="/path/to/nesvor/bin/nesvor" REPO="$WORK_CODE" \
bash "$WORK_CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh" \
  "$FIT_RUN" "$RUN_ROOT/preprocessed" "$SUBJECT" "$RUN_ROOT/prepared" matlab
~~~

该步骤先用 SCMR-v5 MATLAB matcher 生成 mapping，再生成 native cache，最后生成 L100 和所有稀疏 case 的 2D-fit-first 结果。主要输出为：

~~~text
$RUN_ROOT/mapping/scmr_v5/$SUBJECT/
$RUN_ROOT/native_cache/${{SUBJECT}}_scmr_v5/
$RUN_ROOT/reconstruction/fit_first/$SUBJECT/<CASE>/
~~~

检查每个 native-cache manifest 中 source observations、source preprocessed、mapping、native maps 的 SHA256；任何 source SHA 不一致都不能继续评估。

### 5. 服务器：Proposed

Proposed 输入是 sparse observations，不是 fitting map。它可以在 fit-first 之后运行，但不读取 fit-first mapping。下面的命令对 `$RUN_ROOT/sparse_inputs/$SUBJECT/` 下的每个 case 执行 Proposed，并为每个 case 写入 provenance：

~~~bash
DECODER="$ARCHIVE/01_execution_code/01_model_assets/approved_decoder.pth"
B6_MODEL="$ARCHIVE/01_execution_code/01_model_assets/b6_model.pt"
mkdir -p "$RUN_ROOT/reconstruction/proposed/$SUBJECT"

for CASE_ROOT in "$RUN_ROOT/sparse_inputs/$SUBJECT"/*; do
  [ -d "$CASE_ROOT" ] || continue
  CASE=$(basename "$CASE_ROOT")
  SPARSE_MANIFEST="$CASE_ROOT/sparse_manifest.json"
  OUTPUT="$RUN_ROOT/reconstruction/proposed/$SUBJECT/$CASE"
  if [ "$CASE" = "L100_fold0" ]; then
    INPUT_ROOT="$RUN_ROOT/prepared/$SUBJECT"
  else
    INPUT_ROOT="$CASE_ROOT"
  fi
  test -f "$SPARSE_MANIFEST"
  test ! -e "$OUTPUT"

  "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.proposed \
    --prepared-root "$INPUT_ROOT" \
    --subject-id "$SUBJECT" \
    --decoder "$DECODER" \
    --b6-model "$B6_MODEL" \
    --output "$OUTPUT" \
    --execute

  "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.run_provenance \
    --output "$OUTPUT" \
    --method proposed \
    --subject "$SUBJECT" \
    --prepared-root "$INPUT_ROOT" \
    --decoder "$DECODER" \
    --b6-model "$B6_MODEL"
done
~~~

每个 case 的 `config_resolved.yaml`、`experiment_manifest.json`、`run_provenance.json` 和 `sparse_manifest.json` 必须相互对应，并记录 decoder、B6、protocol、sparse selection 和全部 expected SHA256。不得使用历史默认 protocol，也不得把另一个 case 的 manifest 或 decoder 混入当前 case。

### 6. 服务器：final pose 与评估

~~~
ROI_BUNDLE="$RUN_ROOT/roi_manual"
CASES=(L100_fold0 L25_fold0 L50_fold0 \
  SAX100_2CH0_4CH0 SAX100_2CH0_4CH100 SAX100_2CH100_4CH0 \
  S2CH100_4CH50_fold0 S2CH50_4CH100_fold0)

bash "$WORK_CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/paired_final_pose.sh" \
  "$RUN_ROOT" "$RUN_ROOT" "$ROI_BUNDLE" "$SUBJECT" "${{CASES[@]}}"

bash "$WORK_CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/common_native_metrics_qc.sh" \
  "$RUN_ROOT" "$RUN_ROOT" "$ROI_BUNDLE" "$SUBJECT" "${{CASES[@]}}"

OUTPUT_NAME="all_stack_roi_metrics_v2_blood_pool_zero_audit" \
bash "$WORK_CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/common_native_all_stack_roi_metrics.sh" \
  "$RUN_ROOT" "$RUN_ROOT" "$ROI_BUNDLE" "$SUBJECT" "${{CASES[@]}}"
~~~

这些命令只做 final-pose bundle、AHA、LV/心肌/血池指标和 QC，不重新训练或重建。心肌保留 `myocardium_core_legacy` 与 `myocardium_core_1px`；血池为 `inner_filled AND outer_filled`；LV 为 `outer_filled`。评估必须使用 final pose，保留 all-slice、held-out、stack 分层、ROI 像素数和 zero audit。

## {subject} 已归档实例

prepared 输入位于 03_subject_examples/{subject}/01_inputs/prepared；ROI 位于 03_subject_examples/{subject}/01_inputs/roi；两种重建与评估位于 03_subject_examples/{subject}/02_outputs。这是核验和复现示例，不是新 subject 的输出位置。

## 完整性核对

使用 99_integrity/file_sha256.jsonl 逐文件核对复制品 SHA。99_integrity/archive_manifest.json 记录复制计划、代码快照与 subject 示例。任何重跑结果应另存于新的 run root，再与此归档对比。
"""


def _en(subject: str, manifest: dict[str, Any], execution_commit: str) -> str:
    return f"""# SCMR-v5 MultiMap Reproducibility Archive (English)

This is a physically copied, SHA256-recorded reproducibility archive. It contains {subject} as an example subject, while executable entry points take an explicit SUBJECT argument and are generic. Treat the archive as read-only evidence: every rerun must write to a new work root, never back into this directory.

## Layout

~~~
00_live_route_freeze/                 frozen evidence for the successful live route
01_code_snapshot/source.tar           recoverable code snapshot
01_code_snapshot/source.bundle        Git-history bundle
01_execution_code/code/               visible, uncompressed project source tree
01_execution_code/00_pipeline_code_map/ source index organized by pipeline stage
02_runtime_configuration/             separately hashed generic configuration and example provenance
03_subject_examples/{subject}/01_inputs/
  raw_dicom/ deployment_manifest.json preprocessed/ prepared/ roi/ sparse_inputs/
03_subject_examples/{subject}/02_outputs/
  mapping/ native_cache/ fit_first/ proposed/ final_pose/ common_native/
99_integrity/                         archive manifest and per-file SHA256
~~~

The frozen-evidence code commit is {manifest["code_snapshot"]["commit"]}; the visible executable code-tree commit is {execution_commit}. {manifest["file_count"]} copied files are recorded.

## Generic operation, from raw DICOM

### 0. Extract the frozen code and create a generic protocol filename

`01_code_snapshot/` is the read-only evidence snapshot. Use the visible source tree in `01_execution_code/code/` for operations, but first copy it to a new work directory; never edit code within this archive. `02_runtime_configuration/00_common/` holds the subject-agnostic protocol and Proposed configuration; its per-file SHA256 records are in `runtime_configuration_manifest.jsonl`.

~~~bash
ARCHIVE="/path/to/01_reproducible_pipeline"
WORK_CODE="/path/to/new_work/Code_10w"
mkdir -p "$WORK_CODE"
cp -a "$ARCHIVE/01_execution_code/code/." "$WORK_CODE/"
cp "$ARCHIVE/02_runtime_configuration/00_common/protocol_scmr_v5_10_100.yaml" \
   "$WORK_CODE/mlp/configs/protocol_scmr_v5_10_100.yaml"
PROTOCOL="$WORK_CODE/mlp/configs/protocol_scmr_v5_10_100.yaml"
sha256sum "$PROTOCOL"
~~~

### Server run-root layout

Each new subject uses one new, empty `$RUN_ROOT`; every generated file stays there. These names are directly compatible with the generic launchers:

~~~text
$RUN_ROOT/
  config/                 single-subject deployment manifest
  preprocessed/<SUBJECT>/ server MATLAB MAT and QC
  prepared/<SUBJECT>/     observations.npz, timing, bridge manifest/QC
  sparse_inputs/<SUBJECT>/sparse profiles shared by Proposed and fit-first
  mapping/scmr_v5/        MATLAB fitting maps
  native_cache/           SHA-bound native T1/T2/B1 cache
  reconstruction/
    fit_first/<SUBJECT>/  2D-fit-first outputs
    proposed/<SUBJECT>/   Proposed outputs
  roi_manual/             manual ROI bundle uploaded from the local workstation
  evaluation/final_pose/  final pose, AHA, metrics, and QC images
  01_server_preprocess.log
  02_preprocessed_validation.log
  03_prepare_observations.log
  server_matlab_version.txt
~~~

### 1. Server-only: raw DICOM and a single-subject manifest

All computation for a new subject runs on the server. The server must retain the original `sax`, `2ch`, and `4ch` DICOM directories, with exactly ten weights per slice/group. NIfTI cannot replace DICOM because UID, acquisition timing, and DICOM geometry are required. If DICOM initially exists only on a local workstation, the only local data operation is a one-time upload of the raw DICOM; do not run MATLAB or Python locally afterward.

Copy `configs/subject_stack_manifest.example.json` outside the repository. Its historical `local_dicom_dir` field remains, but in server-only operation both `local_dicom_dir` and `server_dicom_dir` must point to the same server-resident DICOM directory. The manifest contains exactly one subject.

~~~bash
SUBJECT="NEW_SUBJECT"
PY="/path/to/server/python"
RUN_ROOT="/path/to/new_server_run/$SUBJECT"
MANIFEST="$RUN_ROOT/config/$SUBJECT-deployment.json"
SERVER_DICOM_ROOT="/path/to/server_dicom"
mkdir -p "$RUN_ROOT/config"
cp "$WORK_CODE/configs/subject_stack_manifest.example.json" "$MANIFEST"
~~~

The only local human task is manual ROI drawing. After drawing, upload the resulting bundle to `$RUN_ROOT/roi_manual/`; it must at least contain `sax_myocardium_masks.npz` and `roi_derivation_manifest.json`. It is consumed only by final-pose evaluation, not by either reconstruction.

### 2. Server-only: DICOM → `preprocessed.mat` → `observations.npz`

Record the server MATLAB environment, then use the launcher below. It runs crop, MIND, and MP-PCA through MATLAB, followed by the Python bridge. `local_preprocess_all.m` is a historical function name; the launcher executes it on the server. No MAT transfer occurs. Outputs are `$RUN_ROOT/preprocessed/`, `$RUN_ROOT/prepared/`, and three run logs.

~~~bash
SERVER_MATLAB_REPORT="$RUN_ROOT/server_matlab_version.txt"
matlab -batch "disp(version); ver" > "$SERVER_MATLAB_REPORT" 2>&1

bash "$WORK_CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/server_preprocess_and_prepare.sh" \
  "$RUN_ROOT" "$SUBJECT" "$MANIFEST" "$PROTOCOL" matlab "$PY"
~~~

Confirm all three stacks have `preprocessed.mat`, `preprocess_qc.json`, `observations.npz`, and `manifest.json`; `Mag_crop` must have ten weights, with MP-PCA enabled and `center_data=true`. The MAT retains truthful historical acquisition/preprocessing metadata `TI=[50,150]`; matching and observation timing deliberately use `TI=[10,100]` through `$PROTOCOL`, while TR/VPS remain sourced from DICOM/MAT.

### 3. Server-only: build sparse case inputs

This Python step derives all standard cases from the server-resident full `observations.npz` and records retained/held-out groups and SHA bindings. It never changes the full observations. Outputs go to `$RUN_ROOT/sparse_inputs/$SUBJECT/` and are shared by Proposed and 2D-fit-first.

~~~bash
bash "$WORK_CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/build_sparse_profiles.sh" \
  "$RUN_ROOT/prepared" "$SUBJECT" "$RUN_ROOT/sparse_inputs" "$PY"
~~~

### 4. Server: corrected SCMR-v5 mapping and 2D-fit-first

Inputs: three `preprocessed.mat` files, prepared observations, and copied sparse inputs. Outputs: mapping MAT files, native-cache manifests, and `T1_3D.nii.gz`/`T2_3D.nii.gz` for every case.

~~~bash
FIT_RUN="$RUN_ROOT"
SPARSE_SOURCE="/path/to/proposed_or_sparse_input_run"
mkdir -p "$FIT_RUN"
cp -a "$SPARSE_SOURCE/sparse_inputs" "$FIT_RUN/"

PY="$PY" NESVOR="/path/to/nesvor/bin/nesvor" REPO="$WORK_CODE" \
bash "$WORK_CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh" \
  "$FIT_RUN" "$RUN_ROOT/preprocessed" "$SUBJECT" "$RUN_ROOT/prepared" matlab
~~~

This first runs the SCMR-v5 MATLAB matcher, then creates the native cache and finally produces L100 and all sparse-case 2D-fit-first outputs. Main outputs are:

~~~text
$RUN_ROOT/mapping/scmr_v5/$SUBJECT/
$RUN_ROOT/native_cache/${{SUBJECT}}_scmr_v5/
$RUN_ROOT/reconstruction/fit_first/$SUBJECT/<CASE>/
~~~

For every native-cache manifest, verify the SHA256 bindings for source observations, source preprocessed MAT, SCMR-v5 mapping, and native maps before accepting a reconstruction.

### 5. Server: Proposed reconstruction

The Proposed method consumes sparse original observations, not a fitting map. The following command runs Proposed for every sparse case and records provenance:

~~~bash
DECODER="$ARCHIVE/01_execution_code/01_model_assets/approved_decoder.pth"
B6_MODEL="$ARCHIVE/01_execution_code/01_model_assets/b6_model.pt"
mkdir -p "$RUN_ROOT/reconstruction/proposed/$SUBJECT"

for CASE_ROOT in "$RUN_ROOT/sparse_inputs/$SUBJECT"/*; do
  [ -d "$CASE_ROOT" ] || continue
  CASE=$(basename "$CASE_ROOT")
  SPARSE_MANIFEST="$CASE_ROOT/sparse_manifest.json"
  OUTPUT="$RUN_ROOT/reconstruction/proposed/$SUBJECT/$CASE"
  if [ "$CASE" = "L100_fold0" ]; then
    INPUT_ROOT="$RUN_ROOT/prepared/$SUBJECT"
  else
    INPUT_ROOT="$CASE_ROOT"
  fi
  test -f "$SPARSE_MANIFEST"
  test ! -e "$OUTPUT"

  "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.proposed \
    --prepared-root "$INPUT_ROOT" \
    --subject-id "$SUBJECT" \
    --decoder "$DECODER" \
    --b6-model "$B6_MODEL" \
    --output "$OUTPUT" \
    --execute

  "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.run_provenance \
    --output "$OUTPUT" \
    --method proposed \
    --subject "$SUBJECT" \
    --prepared-root "$INPUT_ROOT" \
    --decoder "$DECODER" \
    --b6-model "$B6_MODEL"
done
~~~

For each case, use only that case's `config_resolved.yaml`, `experiment_manifest.json`, `run_provenance.json`, and `sparse_manifest.json` to obtain the decoder, B6, protocol, sparse selection, and expected SHA256 values. Do not substitute the historical/default protocol or another case's decoder.

### 6. Server: final pose, metrics, and visual QC

Input: completed Proposed and fit-first outputs plus the manual ROI bundle. Output: final-pose bundles, AHA results, LV/myocardium/blood-pool metrics, mask overlays, slice-pair QC, and all-stack statistics.

~~~bash
ROI_BUNDLE="$RUN_ROOT/roi_manual"
CASES=(L100_fold0 L25_fold0 L50_fold0 \
  SAX100_2CH0_4CH0 SAX100_2CH0_4CH100 SAX100_2CH100_4CH0 \
  S2CH100_4CH50_fold0 S2CH50_4CH100_fold0)

bash "$WORK_CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/paired_final_pose.sh" \
  "$RUN_ROOT" "$RUN_ROOT" "$ROI_BUNDLE" "$SUBJECT" "${{CASES[@]}}"
bash "$WORK_CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/common_native_metrics_qc.sh" \
  "$RUN_ROOT" "$RUN_ROOT" "$ROI_BUNDLE" "$SUBJECT" "${{CASES[@]}}"
OUTPUT_NAME="all_stack_roi_metrics_v2_blood_pool_zero_audit" \
bash "$WORK_CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/common_native_all_stack_roi_metrics.sh" \
  "$RUN_ROOT" "$RUN_ROOT" "$ROI_BUNDLE" "$SUBJECT" "${{CASES[@]}}"
~~~

These commands only create final-pose bundles, AHA, LV/myocardium/blood-pool metrics, and QC; they do not retrain or reconstruct. Keep `myocardium_core_legacy` and `myocardium_core_1px`; blood pool is `inner_filled AND outer_filled`, and LV is `outer_filled`. All quantitative metrics must use final pose and retain all-slice, held-out, stack strata, ROI voxel counts, mean/SD, full metrics, and zero-value audit.

## Archived {subject} example

- Prepared input: 03_subject_examples/{subject}/01_inputs/prepared
- ROI: 03_subject_examples/{subject}/01_inputs/roi
- Reconstruction and evaluation outputs: 03_subject_examples/{subject}/02_outputs

## Integrity

Verify copied files with 99_integrity/file_sha256.jsonl. 99_integrity/archive_manifest.json records the copy plan, code snapshot, and subject example. Rerun outputs belong in a new work root and should be compared against this archive.
"""


def build_archive_readmes(archive_root: str | Path, *, replace_existing: bool = False) -> tuple[Path, Path]:
    archive = Path(archive_root)
    manifest = _load(archive / "99_integrity" / "archive_manifest.json")
    if manifest.get("schema_version") != "scmr_v5_multimap_reproducibility_archive/v1":
        raise ValueError("not a completed SCMR-v5 reproducibility archive")
    subjects = manifest.get("subject_examples")
    if not isinstance(subjects, list) or len(subjects) != 1 or not isinstance(subjects[0], str):
        raise ValueError("archive README requires exactly one subject example")
    for required in ("00_live_route_freeze", "01_code_snapshot", "01_execution_code", "02_runtime_configuration", "03_subject_examples", "99_integrity"):
        if not (archive / required).is_dir():
            raise FileNotFoundError(f"archive layout is incomplete: {required}")
    execution_metadata = archive / "01_execution_code" / "99_integrity" / "execution_metadata.json"
    execution_commit = "not-recorded"
    if execution_metadata.is_file():
        value = _load(execution_metadata).get("code_commit")
        if isinstance(value, str) and value:
            execution_commit = value
    zh, en = archive / "README_zh-CN.md", archive / "README_en.md"
    if (zh.exists() or en.exists()) and not replace_existing:
        raise FileExistsError("refusing to overwrite existing archive README")
    zh.write_text(_zh(subjects[0], manifest, execution_commit), encoding="utf-8")
    en.write_text(_en(subjects[0], manifest, execution_commit), encoding="utf-8")
    return zh, en


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--replace-existing", action="store_true", help="replace only the two archive README files")
    args = parser.parse_args()
    for path in build_archive_readmes(args.archive_root, replace_existing=args.replace_existing):
        print(path)


if __name__ == "__main__":
    main()
