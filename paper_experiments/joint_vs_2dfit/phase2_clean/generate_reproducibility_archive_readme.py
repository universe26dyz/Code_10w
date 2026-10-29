"""Generate bilingual operation manuals after a reproducibility archive is complete."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _zh(subject: str, manifest: dict[str, Any]) -> str:
    return f"""# SCMR-v5 MultiMap 可复现实验归档（中文）

本目录是已经完成物理复制与 SHA256 登记的可复现实验归档。它保存一个实际 subject 示例（{subject}），但软件入口均使用显式 SUBJECT 参数，适用于其他 subject。归档本身是只读证据；所有重新运行都必须写到新的工作目录，绝不能把新结果写回这里。

## 目录用途

~~~
00_live_route_freeze/                 原始分散成功路线的冻结证据
01_code_snapshot/source.tar           可恢复的代码快照
01_code_snapshot/source.bundle        Git 历史 bundle
03_subject_examples/{subject}/01_inputs/
  raw_dicom/ deployment_manifest.json preprocessed/ prepared/ roi/ sparse_inputs/
03_subject_examples/{subject}/02_outputs/
  mapping/ native_cache/ fit_first/ proposed/ final_pose/ common_native/
99_integrity/                         archive_manifest 与逐文件 SHA256
~~~

本归档的代码 commit 为 {manifest["code_snapshot"]["commit"]}，共登记 {manifest["file_count"]} 个复制文件。

## 从原始 DICOM 开始：通用运行顺序

### 0. 先取出代码，并建立本次运行的通用配置副本

`01_code_snapshot/source.tar` 是本次成功路线的只读代码证据。先解压到新的工作目录；之后的改动不能发生在归档目录内。归档中的正确 protocol 文件具有历史文件名 `protocol_hhz_scmr_v5_10_100.yaml`，但内容并不依赖 HHZ。将其复制为本次工作目录中的通用名称，并始终把这个新路径显式传入下游命令。

~~~bash
ARCHIVE="/path/to/01_reproducible_pipeline"
WORK_CODE="/path/to/new_work/Code_10w"
mkdir -p "$WORK_CODE"
tar -xf "$ARCHIVE/01_code_snapshot/source.tar" -C "$WORK_CODE"
cp "$WORK_CODE/mlp/configs/protocol_hhz_scmr_v5_10_100.yaml" \
   "$WORK_CODE/mlp/configs/protocol_scmr_v5_10_100.yaml"
PROTOCOL="$WORK_CODE/mlp/configs/protocol_scmr_v5_10_100.yaml"
sha256sum "$PROTOCOL"
~~~

### 1. 本地原始 DICOM

每个 subject 需要 sax、2ch、4ch 三个 DICOM 目录；每个 slice/group 必须有十个 weight。不要以 NIfTI 代替 DICOM，因为后续步骤需要 SOP UID、采集时间和 DICOM 空间几何。将 configs/subject_stack_manifest.example.json 复制到代码外，填写本地 local_dicom_dir 与服务器中相同 DICOM 的 server_dicom_dir；每次只包含一个 subject。

### 2. 本地 MATLAB：DICOM → preprocessed.mat

~~~
matlab -batch "addpath('$CODE/trad/scripts'); local_preprocess_all('$MANIFEST','$LOCAL_PREPROCESSED_ROOT');"
~~~

输出为每个 stack 的 preprocessed.mat、preprocess_qc.json。检查 Mag_crop 的第三维为 10，且 QC 为 MP-PCA enabled、center_data=true。源 MAT 的历史 TI=[50,150] 不修改；SCMR-v5 timing/matching 使用 TI=[10,100]。

### 3. 服务器：MAT + DICOM → observations.npz

复制 MAT 和单 subject manifest 到服务器，并保留同一批服务器 DICOM。先运行 validate_transferred_preprocessed.py，再运行 prepare_all_observations.py，显式传入上一步建立的 `$PROTOCOL`。observations.npz 是本步骤从 MAT 像素、DICOM UID/几何和正确 protocol 生成的桥接产物，不是原始输入；其 TR/VPS 必须与 MAT 一致。

~~~bash
PY="/path/to/server/python"
SERVER_DEPLOYMENT_MANIFEST="/path/to/server/$SUBJECT-deployment.json"
SERVER_PREPROCESSED_ROOT="/path/to/server_preprocessed"
SERVER_PREPARED_ROOT="/path/to/server_prepared"

"$PY" "$WORK_CODE/trad/scripts/validate_transferred_preprocessed.py" \
  --manifest "$SERVER_DEPLOYMENT_MANIFEST" \
  --preprocessed-root "$SERVER_PREPROCESSED_ROOT" \
  --output "$SERVER_PREPROCESSED_ROOT/$SUBJECT/transfer_qc.json"
"$PY" "$WORK_CODE/trad/scripts/prepare_all_observations.py" \
  --manifest "$SERVER_DEPLOYMENT_MANIFEST" \
  --preprocessed-root "$SERVER_PREPROCESSED_ROOT" \
  --prepared-root "$SERVER_PREPARED_ROOT" \
  --protocol "$PROTOCOL"
~~~

### 4. 服务器：SCMR-v5 fitting 与 2D-fit-first

~~~
bash "$CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh" \
  "$FIT_RUN" "$SERVER_PREPROCESSED_ROOT" "$SUBJECT" "$SERVER_PREPARED_ROOT" matlab
~~~

检查每个 native-cache manifest 中 source observations、source preprocessed、mapping、native maps 的 SHA256。

### 5. 服务器：Proposed

Proposed 输入是 sparse observations，不是 fitting map。对每个 case，从同一 case 的 config_resolved.yaml、experiment_manifest.json、run_provenance.json 读取 decoder、B6、protocol、sparse manifest 和全部 expected SHA256，并显式传给 phase2_clean.proposed --execute。不得使用历史默认 protocol。

### 6. 服务器：final pose 与评估

~~~
bash "$CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/paired_final_pose.sh" \
  "$PROPOSED_RUN" "$FIT_RUN" "$ROI_BUNDLE" "$SUBJECT" "$CASE"
bash "$CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/common_native_metrics_qc.sh" \
  "$PROPOSED_RUN" "$FIT_RUN" "$ROI_BUNDLE" "$SUBJECT" "$CASE"
OUTPUT_NAME="all_stack_roi_metrics_v2_blood_pool_zero_audit" \
bash "$CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/common_native_all_stack_roi_metrics.sh" \
  "$PROPOSED_RUN" "$FIT_RUN" "$ROI_BUNDLE" "$SUBJECT" "$CASE"
~~~

心肌保留 myocardium_core_legacy 与 myocardium_core_1px；血池为 inner_filled AND outer_filled；LV 为 outer_filled。评估必须使用 final pose，保留 all-slice、held-out、stack 分层、ROI 像素数和 zero audit。

## {subject} 已归档实例

prepared 输入位于 03_subject_examples/{subject}/01_inputs/prepared；ROI 位于 03_subject_examples/{subject}/01_inputs/roi；两种重建与评估位于 03_subject_examples/{subject}/02_outputs。这是核验和复现示例，不是新 subject 的输出位置。

## 完整性核对

使用 99_integrity/file_sha256.jsonl 逐文件核对复制品 SHA。99_integrity/archive_manifest.json 记录复制计划、代码快照与 subject 示例。任何重跑结果应另存于新的 run root，再与此归档对比。
"""


def _en(subject: str, manifest: dict[str, Any]) -> str:
    return f"""# SCMR-v5 MultiMap Reproducibility Archive (English)

This is a physically copied, SHA256-recorded reproducibility archive. It contains {subject} as an example subject, while executable entry points take an explicit SUBJECT argument and are generic. Treat the archive as read-only evidence: every rerun must write to a new work root, never back into this directory.

## Layout

~~~
00_live_route_freeze/                 frozen evidence for the successful live route
01_code_snapshot/source.tar           recoverable code snapshot
01_code_snapshot/source.bundle        Git-history bundle
03_subject_examples/{subject}/01_inputs/
  raw_dicom/ deployment_manifest.json preprocessed/ prepared/ roi/ sparse_inputs/
03_subject_examples/{subject}/02_outputs/
  mapping/ native_cache/ fit_first/ proposed/ final_pose/ common_native/
99_integrity/                         archive manifest and per-file SHA256
~~~

The archived code commit is {manifest["code_snapshot"]["commit"]}; {manifest["file_count"]} copied files are recorded.

## Generic operation, from raw DICOM

### 0. Extract the frozen code and create a generic protocol filename

`01_code_snapshot/source.tar` is the read-only evidence snapshot. Extract it to a new work directory; never edit code within this archive. The archived protocol has the historical filename `protocol_hhz_scmr_v5_10_100.yaml`, but its contents are the reusable SCMR-v5 10/100 contract. Copy it to a generic filename and pass that filename explicitly below.

~~~bash
ARCHIVE="/path/to/01_reproducible_pipeline"
WORK_CODE="/path/to/new_work/Code_10w"
mkdir -p "$WORK_CODE"
tar -xf "$ARCHIVE/01_code_snapshot/source.tar" -C "$WORK_CODE"
cp "$WORK_CODE/mlp/configs/protocol_hhz_scmr_v5_10_100.yaml" \
   "$WORK_CODE/mlp/configs/protocol_scmr_v5_10_100.yaml"
PROTOCOL="$WORK_CODE/mlp/configs/protocol_scmr_v5_10_100.yaml"
sha256sum "$PROTOCOL"
~~~

### 1. Local machine: raw DICOM to `preprocessed.mat`

Input is the original DICOM for `sax`, `2ch`, and `4ch`, with ten weights per slice/group. NIfTI cannot replace DICOM because UID, acquisition timing, and DICOM geometry are required. Copy `configs/subject_stack_manifest.example.json` outside the repository and fill in `local_dicom_dir` and the matching server `server_dicom_dir` for one subject.

~~~bash
SUBJECT="NEW_SUBJECT"
LOCAL_DICOM_ROOT="/path/to/local_dicom"
DEPLOYMENT_MANIFEST="/path/outside_code/$SUBJECT-deployment.json"
LOCAL_PREPROCESSED_ROOT="/path/to/local_preprocessed"

cp "$WORK_CODE/configs/subject_stack_manifest.example.json" "$DEPLOYMENT_MANIFEST"
matlab -batch "addpath('$WORK_CODE/trad/scripts'); local_preprocess_all('$DEPLOYMENT_MANIFEST','$LOCAL_PREPROCESSED_ROOT');"
~~~

Output is `<LOCAL_PREPROCESSED_ROOT>/<SUBJECT>/<stack>/preprocessed.mat` plus `preprocess_qc.json`. Confirm all three stacks exist, `Mag_crop` has 10 weights, MP-PCA is enabled, and `center_data=true`. The source MAT retains historical acquisition metadata `TI=[50,150]`; do not modify it. Matching and observation timing deliberately use `TI=[10,100]` through `$PROTOCOL`.

### 2. Server: validate transfer and make `observations.npz`

Copy the MATLAB outputs and the single-subject deployment manifest to the server. The server must retain the same raw DICOM specified by `server_dicom_dir`. `observations.npz` is generated here from MAT pixels, DICOM UID/geometry, and the explicit protocol; it is not a raw input.

~~~bash
PY="/path/to/server/python"
SERVER_DEPLOYMENT_MANIFEST="/path/to/server/$SUBJECT-deployment.json"
SERVER_PREPROCESSED_ROOT="/path/to/server_preprocessed"
SERVER_PREPARED_ROOT="/path/to/server_prepared"

"$PY" "$WORK_CODE/trad/scripts/validate_transferred_preprocessed.py" \
  --manifest "$SERVER_DEPLOYMENT_MANIFEST" \
  --preprocessed-root "$SERVER_PREPROCESSED_ROOT" \
  --output "$SERVER_PREPROCESSED_ROOT/$SUBJECT/transfer_qc.json"
"$PY" "$WORK_CODE/trad/scripts/prepare_all_observations.py" \
  --manifest "$SERVER_DEPLOYMENT_MANIFEST" \
  --preprocessed-root "$SERVER_PREPROCESSED_ROOT" \
  --prepared-root "$SERVER_PREPARED_ROOT" \
  --protocol "$PROTOCOL"
~~~

Check `PREPARED_BATCH_QC.json`, the three stack `observations.npz` files, and every prepared manifest. TR/VPS must agree with the MAT/DICOM data; the protocol must state TI 10/100, TR 2.61 ms, VPS 87, normalized absolute 10D inner-product matching, and no interpolation/refinement.

### 3. Server: corrected SCMR-v5 mapping and 2D-fit-first

Inputs: three `preprocessed.mat` files, prepared observations, and copied sparse inputs. Outputs: mapping MAT files, native-cache manifests, and `T1_3D.nii.gz`/`T2_3D.nii.gz` for every case.

~~~bash
FIT_RUN="/path/to/new_fit_run"
SPARSE_SOURCE="/path/to/proposed_or_sparse_input_run"
mkdir -p "$FIT_RUN"
cp -a "$SPARSE_SOURCE/sparse_inputs" "$FIT_RUN/"

bash "$WORK_CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh" \
  "$FIT_RUN" "$SERVER_PREPROCESSED_ROOT" "$SUBJECT" "$SERVER_PREPARED_ROOT" matlab
~~~

For every native-cache manifest, verify the SHA256 bindings for source observations, source preprocessed MAT, SCMR-v5 mapping, and native maps before accepting a reconstruction.

### 4. Server: Proposed reconstruction

The Proposed method consumes sparse original observations, not a fitting map. For each case, use only the same case's `config_resolved.yaml`, `experiment_manifest.json`, `run_provenance.json`, and `sparse_manifest.json` to obtain the decoder, B6, protocol, sparse selection, and expected SHA256 values. Invoke `phase2_clean.proposed --execute` with those explicit values. Do not substitute the historical/default protocol or another case's decoder.

### 5. Server: final pose, metrics, and visual QC

Input: completed Proposed and fit-first outputs plus the manual ROI bundle. Output: final-pose bundles, AHA results, LV/myocardium/blood-pool metrics, mask overlays, slice-pair QC, and all-stack statistics.

~~~bash
PROPOSED_RUN="/path/to/proposed_run"
ROI_BUNDLE="/path/to/manual_roi_bundle"
CASE="L100_fold0"

bash "$WORK_CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/paired_final_pose.sh" \
  "$PROPOSED_RUN" "$FIT_RUN" "$ROI_BUNDLE" "$SUBJECT" "$CASE"
bash "$WORK_CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/common_native_metrics_qc.sh" \
  "$PROPOSED_RUN" "$FIT_RUN" "$ROI_BUNDLE" "$SUBJECT" "$CASE"
OUTPUT_NAME="all_stack_roi_metrics_v2_blood_pool_zero_audit" \
bash "$WORK_CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/common_native_all_stack_roi_metrics.sh" \
  "$PROPOSED_RUN" "$FIT_RUN" "$ROI_BUNDLE" "$SUBJECT" "$CASE"
~~~

Keep `myocardium_core_legacy` and `myocardium_core_1px`; blood pool is `inner_filled AND outer_filled`, and LV is `outer_filled`. All quantitative metrics must use final pose and retain all-slice, held-out, stack strata, ROI voxel counts, mean/SD, full metrics, and zero-value audit.

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
    for required in ("00_live_route_freeze", "01_code_snapshot", "03_subject_examples", "99_integrity"):
        if not (archive / required).is_dir():
            raise FileNotFoundError(f"archive layout is incomplete: {required}")
    zh, en = archive / "README_zh-CN.md", archive / "README_en.md"
    if (zh.exists() or en.exists()) and not replace_existing:
        raise FileExistsError("refusing to overwrite existing archive README")
    zh.write_text(_zh(subjects[0], manifest), encoding="utf-8")
    en.write_text(_en(subjects[0], manifest), encoding="utf-8")
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
