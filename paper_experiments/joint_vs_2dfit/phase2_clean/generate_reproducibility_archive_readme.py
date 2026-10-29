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

### 1. 本地原始 DICOM

每个 subject 需要 sax、2ch、4ch 三个 DICOM 目录；每个 slice/group 必须有十个 weight。不要以 NIfTI 代替 DICOM，因为后续步骤需要 SOP UID、采集时间和 DICOM 空间几何。将 configs/subject_stack_manifest.example.json 复制到代码外，填写本地 local_dicom_dir 与服务器中相同 DICOM 的 server_dicom_dir；每次只包含一个 subject。

### 2. 本地 MATLAB：DICOM → preprocessed.mat

~~~
matlab -batch "addpath('$CODE/trad/scripts'); local_preprocess_all('$MANIFEST','$LOCAL_PREPROCESSED_ROOT');"
~~~

输出为每个 stack 的 preprocessed.mat、preprocess_qc.json。检查 Mag_crop 的第三维为 10，且 QC 为 MP-PCA enabled、center_data=true。源 MAT 的历史 TI=[50,150] 不修改；SCMR-v5 timing/matching 使用 TI=[10,100]。

### 3. 服务器：MAT + DICOM → observations.npz

复制 MAT 和单 subject manifest 到服务器，并保留同一批服务器 DICOM。先运行 validate_transferred_preprocessed.py，再运行 prepare_all_observations.py，显式传入已冻结的 mlp/configs/protocol_hhz_scmr_v5_10_100.yaml。该文件名是历史实验命名；其内容是通用的 SCMR-v5 10/100 protocol，不能因为 subject 改变而替换为旧默认 protocol。observations.npz 是本步骤从 MAT 像素、DICOM UID/几何和正确 protocol 生成的桥接产物，不是原始输入；其 TR/VPS 必须与 MAT 一致。

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

1. **Local raw DICOM:** supply sax, 2ch, and 4ch, with ten weights per slice/group. NIfTI cannot replace DICOM because UID, acquisition timing, and DICOM geometry are required.
2. **Local MATLAB:** create preprocessed.mat and preprocess_qc.json with local_preprocess_all. Verify 10 weights, MP-PCA enabled, and center_data=true.
3. **Server preparation:** copy MAT and the single-subject deployment manifest; retain the same raw DICOM on the server. Validate MAT, then run prepare_all_observations.py with the frozen mlp/configs/protocol_hhz_scmr_v5_10_100.yaml. Its filename is historical; its SCMR-v5 10/100 contents are the reusable contract. observations.npz is generated here, not supplied as raw input.
4. **Generic fit-first:** run server_commands/run_scmr_v5_fit_first.sh with FIT_RUN, PREPROCESSED_ROOT, SUBJECT, PREPARED_ROOT, and MATLAB. Verify every native-cache SHA binding.
5. **Proposed:** use sparse observations and explicit decoder, B6, protocol, and expected SHA256 values from the same case provenance. Do not use the historical default protocol.
6. **Evaluation:** use paired_final_pose.sh, common_native_metrics_qc.sh, and common_native_all_stack_roi_metrics.sh, all with an explicit subject. Preserve both myocardial cores, LV, blood pool, all-slice/held-out/stack strata, and zero audit.

## Archived {subject} example

- Prepared input: 03_subject_examples/{subject}/01_inputs/prepared
- ROI: 03_subject_examples/{subject}/01_inputs/roi
- Reconstruction and evaluation outputs: 03_subject_examples/{subject}/02_outputs

## Integrity

Verify copied files with 99_integrity/file_sha256.jsonl. 99_integrity/archive_manifest.json records the copy plan, code snapshot, and subject example. Rerun outputs belong in a new work root and should be compared against this archive.
"""


def build_archive_readmes(archive_root: str | Path) -> tuple[Path, Path]:
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
    if zh.exists() or en.exists():
        raise FileExistsError("refusing to overwrite existing archive README")
    zh.write_text(_zh(subjects[0], manifest), encoding="utf-8")
    en.write_text(_en(subjects[0], manifest), encoding="utf-8")
    return zh, en


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True)
    for path in build_archive_readmes(parser.parse_args().archive_root):
        print(path)


if __name__ == "__main__":
    main()
