"""Generate bilingual human-readable records for a frozen live pipeline route."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _case_table(report: dict[str, Any]) -> str:
    rows = [
        "| Case | sparse manifest | Proposed model | Fit-first T1 | final-pose metrics | common-native metrics |",
        "|---|---|---|---|---|---|",
    ]
    for case, values in report["per_case"].items():
        proposed, fit_first, evaluation = values["proposed"], values["fit_first"], values["evaluation"]
        cells = (
            proposed.get("sparse_manifest", {}).get("sha256", ""),
            proposed.get("proposed_model", {}).get("sha256", ""),
            fit_first.get("fit_first_t1", {}).get("sha256", ""),
            evaluation.get("final_pose_metrics", {}).get("sha256", ""),
            evaluation.get("common_native_metrics", {}).get("sha256", ""),
        )
        rows.append("| " + case + " | " + " | ".join(_text(cell) for cell in cells) + " |")
    return "\n".join(rows)


def _native_table(report: dict[str, Any]) -> str:
    rows = [
        "| Stack | method | source observations SHA256 | source preprocessed SHA256 | native map SHA256 |",
        "|---|---|---|---|---|",
    ]
    for stack, manifest in report["native_cache_manifests"].items():
        rows.append("| " + " | ".join((
            stack,
            _text(manifest.get("method")),
            _text(manifest.get("source_observations", {}).get("sha256")),
            _text(manifest.get("source_preprocessed", {}).get("sha256")),
            _text(manifest.get("native_maps_sha256")),
        )) + " |")
    return "\n".join(rows)


def _case_artifact_details(report: dict[str, Any]) -> str:
    lines: list[str] = []
    for case, values in report["per_case"].items():
        lines.append(f"### `{case}`")
        for category in ("proposed", "fit_first", "evaluation"):
            for name, item in values[category].items():
                path, digest = item.get("path"), item.get("sha256")
                if path and digest:
                    lines.append(f"- `{name}`: `{path}`  ")
                    lines.append(f"  SHA256: `{digest}`")
        lines.append("")
    return "\n".join(lines)


def _commands(report: dict[str, Any]) -> dict[str, str]:
    """Derive live-route command names from evidence, without subject hardcoding."""
    subject = report["subject"]
    roots = report["roots"]
    proposed_subject = Path(roots["proposed_reconstructions"])
    fit_subject = Path(roots["fit_first_reconstructions"])
    source_preprocessed = report.get("native_cache_manifests", {}).get("sax", {}).get("source_preprocessed", {}).get("path")

    def run_root(subject_root: Path) -> Path:
        for parent in subject_root.parents:
            if parent.name == "reconstruction":
                return parent.parent
        return subject_root.parent

    return {
        "subject": subject,
        "subject_script": subject.lower(),
        "proposed_run": str(run_root(proposed_subject)),
        "fit_run": str(run_root(fit_subject)),
        "prepared_root": str(Path(roots["shared_prepared_observations"]).parent),
        "preprocessed_root": str(Path(source_preprocessed).parents[2]) if source_preprocessed else "/path/to/preprocessed_root",
        "roi": roots["shared_manual_roi"],
        "code": report["code"]["path"],
    }


def _zh_readme(report: dict[str, Any], semantic: dict[str, Any]) -> str:
    command = _commands(report)
    subject = command["subject"]
    roots = report["roots"]
    protocol = semantic["protocol"]
    approved_cases = [case for case, value in semantic["cases"].items() if value.get("status") == "PASS"]
    return f"""# SCMR-v5 MultiMap 已冻结现场路线记录（中文）

这是一份**给新操作者照着执行的现场路线手册**：先按下面步骤了解并复核已经跑通的路线，最后再到附录查实际 `{subject}` 文件路径、SHA256 与冻结版本。它记录的是仍在原位置的成功路线；任何原始输入和输出都不得移动、重命名、覆盖或修改。

软件、config 和正式 launcher 必须保持 subject 通用；`{subject}` 仅是本冻结记录中的实例数据。

## 操作步骤

### 0. 先确认工作地点与边界

| 工作 | 在哪里运行 | 输入 | 输出 | 目的 |
|---|---|---|---|---|
| 检查代码、版本与 SHA | 本地或服务器 code checkout | 冻结目录与原文件 | 检查结果 | 确认没有把旧路线混入。 |
| SCMR-v5 MATLAB fitting | 服务器（MATLAB） | `preprocessed.mat` | 三个 stack 的 mapping 与 native cache | 获得正确的 native T1/T2/B1。 |
| 2D-fit-first / NeSVoR | 服务器 GPU | native cache 与 sparse input | T1/T2 volume | 重建 fit-first 对照方法。 |
| Proposed / MLP | 服务器 GPU | sparse input、冻结 decoder/B6/protocol | checkpoint 与 T1/T2 volume | 重建 Proposed 方法。 |
| final pose 与定量评估 | 服务器 | 两种方法的 volume、ROI | 指标 CSV/JSON、AHA、QC 图 | 在统一 pose/ROI 下比较。 |

下面所有会产生文件的命令都应写到**新的空运行根**；不要对本附录记录的完成目录再次运行 reconstruction。评估可只读现有重建结果。

### 1. 在服务器设置变量并做最小预检

输入：服务器代码 checkout、prepared observations、ROI、两种方法的运行根。输出：仅终端检查结果，不写入数据。

```bash
CODE="{command['code']}"
SUBJECT="{subject}"
PROPOSED_RUN="{command['proposed_run']}"
FIT_RUN="{command['fit_run']}"
PREPARED_ROOT="{command['prepared_root']}"
ROI_BUNDLE="{command['roi']}"

test -d "$CODE" && test -d "$PREPARED_ROOT/$SUBJECT" && test -d "$ROI_BUNDLE"
git -C "$CODE" rev-parse HEAD
```

完成判据：目录存在，Git commit 与底部“冻结现场文件与 SHA 附录”一致。若不一致，停止；不要用目录名猜测版本。

### 2. 核对共同输入与稀疏方案

输入：`$PREPARED_ROOT/$SUBJECT/{{sax,2ch,4ch}}/observations.npz`、每个 case 的 `sparse_manifest.json`。输出：文件 SHA 与 manifest 内容的比对结果。

```bash
for STACK in sax 2ch 4ch; do
  sha256sum "$PREPARED_ROOT/$SUBJECT/$STACK/observations.npz"
done

find "$PROPOSED_RUN/sparse_inputs/$SUBJECT" -name sparse_manifest.json -print
```

作用：prepared observations 是两种方法共同的原始 10-weight 信号；`sparse_manifest.json` 才是每个 case 的 retained/held-out group 的权威来源。完成判据：三个 observations SHA、case 名称和附录一致。不得把旧 `ismrm_v1` 或 generic/TRAD cache 当作输入。

### 3. 在服务器生成正确的 SCMR-v5 mapping 与 fit-first 重建

输入：三个 `preprocessed.mat`、prepared observations、已复制到新根的 sparse inputs。输出：`mapping/scmr_v5/$SUBJECT/*_mapping.mat`、`native_cache/${{SUBJECT}}_scmr_v5`、`reconstruction/fit_first/$SUBJECT/<CASE>/`。

先为一次新的可复跑实验建立空根并复制 sparse input（复制，不移动现有文件）：

```bash
NEW_FIT_RUN="/path/to/new_run"
mkdir -p "$NEW_FIT_RUN"
cp -a "$PROPOSED_RUN/sparse_inputs" "$NEW_FIT_RUN/"

bash "$CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_{command['subject_script']}_scmr_v5_fit_first.sh" \\
  "$NEW_FIT_RUN" \\
  "{command['preprocessed_root']}" \\
  "$PREPARED_ROOT" \\
  matlab
```

作用：脚本先以 SCMR-v5 MATLAB matcher 生成 mapping，再构造 native cache，随后对 sparse cases 做 fit-first/NeSVoR。完成判据：每个 stack 的 `manifest.json` 同时记录 `source_observations.sha256`、`source_preprocessed.sha256`、`scmr_v5_mapping.sha256`、`native_maps_sha256`；每个 case 有 `input_manifest.json`、`commands.sh` 和 T1/T2 `volume_ms.nii.gz`。本次正确 contract 是 `TI=[10,100] ms`、`TR={protocol['tr_ms']} ms`、`VPS={protocol['vps']}`，而不是历史 `TI=[50,150]`。

### 4. 在服务器运行或复核 Proposed

输入：该 case 的 sparse input、冻结 decoder/B6、正确 protocol、`config_resolved.yaml`。输出：`model.pt`、`T1_3D.nii.gz`、`T2_3D.nii.gz`、`experiment_manifest.json`、`run_provenance.json`。

执行前先从**同一 case**的 `config_resolved.yaml`、`experiment_manifest.json` 和 `run_provenance.json` 取 decoder、B6、protocol 与期望 SHA；不要依赖脚本默认 protocol。运行完成后执行：

```bash
CASE="L100_fold0"  # 改为要检查的 case
find "$PROPOSED_RUN/reconstruction/proposed/$SUBJECT/$CASE" -maxdepth 1 \\
  \\( -name model.pt -o -name T1_3D.nii.gz -o -name T2_3D.nii.gz -o \\
     -name config_resolved.yaml -o -name experiment_manifest.json -o -name run_provenance.json \\) -print
```

完成判据：六类文件齐全，且附录的 case SHA 及 semantic verification 都为 PASS。Proposed 使用原始信号；不可用错误 mapping cache 代替它的输入。

### 5. 在服务器运行 final-pose、AHA 与心肌/LV 定量

输入：已完成的 Proposed root、已完成的 fit-first root、同一 ROI bundle。输出：final-pose bundle、AHA（legacy 与 1px）、心肌/LV 指标、common-native all-stack/held-out/blood-pool/zero-audit 结果和 QC 图。

```bash
CASE="L100_fold0"  # 可替换为一个或多个已完成 case
bash "$CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/{command['subject_script']}_paired_final_pose.sh" \\
  "$PROPOSED_RUN" "$FIT_RUN" "$ROI_BUNDLE" "$CASE"

bash "$CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/{command['subject_script']}_common_native_metrics_qc.sh" \\
  "$PROPOSED_RUN" "$FIT_RUN" "$ROI_BUNDLE" "$CASE"

OUTPUT_NAME="all_stack_roi_metrics_v2_blood_pool_zero_audit" \\
bash "$CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/{command['subject_script']}_common_native_all_stack_roi_metrics.sh" \\
  "$PROPOSED_RUN" "$FIT_RUN" "$ROI_BUNDLE" "$CASE"
```

作用：final-pose 将两种方法投到同一 native 查询面；心肌定量保留 `myocardium_core_1px` 与 `myocardium_core_legacy`，血池固定为 `inner_filled AND outer_filled`，LV 为 `outer_filled`。完成判据：输出 manifest 写明 final pose、mask、ROI 像素数、all-slice/held-out 分层以及 zero audit。数值为 0 的 fit-first voxel 不会被自动排除，必须查看 zero-audit。

### 6. 最后做可复现性检查

输入：本次产生的 manifest 和冻结目录。输出：SHA 比对记录。

```bash
sha256sum <file>
grep -F '"path": "<original-path>"' "<freeze_root>/source_file_manifest.jsonl"
```

只有当原文件与复制文件 SHA256 一致、所有 Proposed case 的 semantic verification 为 PASS、各 stack 的 native-cache provenance 一致时，才可声明新副本可复现。不要把 3D reconstructed volume 用作 AHA native 2D reference，不要转置 SAX ROI（均为 `[group,row,col]`）。

## 冻结现场文件与 SHA 附录

### 冻结状态

- 冻结代码 Git commit：`{report['code']['git_head']}`
- 代码根目录：`{report['code']['path']}`
- Git 状态：`code/git_status.txt`
- 已跟踪改动 patch：`code/git_worktree.patch`
- Git bundle：`code/code_head.bundle`
- 当前 subject：`{subject}`
- semantic verification 通过的 case：{', '.join(approved_cases)}
- 正确 protocol：`{protocol['path']}`（SHA256：`{protocol['sha256']}`）
- fitting timing：`delta_AcquisitionTime - TR*(EchoTrainLength+10) - preparation_delay`；normalized absolute 10D matching、QC 0.90、无 interpolation/refinement。

### 当前现场路径

| 类别 | 路径 | 作用 |
|---|---|---|
| prepared observations | `{roots['shared_prepared_observations']}` | 三个 stack 的已准备 observations。 |
| manual ROI | `{roots['shared_manual_roi']}` | outer/inner filled、血池与心肌 ROI。 |
| sparse inputs | `{roots['shared_sparse_inputs']}` | 每个 case 的 sparse manifest 与 retained observations。 |
| MATLAB mapping | `{roots['fit_first_matlab_mapping']}` | 修正参数后的 mapping。 |
| SCMR-v5 native cache | `{roots['fit_first_native_cache']}` | fit-first/native reference 权威 cache。 |
| fit-first outputs | `{roots['fit_first_reconstructions']}` | NeSVoR 输出。 |
| Proposed outputs | `{roots['proposed_reconstructions']}` | MLP 输出与 provenance。 |
| final-pose evaluation | `{roots['final_pose_evaluation']}` | AHA、心肌/LV metrics、QC。 |
| common-native evaluation | `{roots['common_native_evaluation']}` | all-stack、held-out、blood-pool、zero audit。 |

### SCMR-v5 native-cache SHA

{_native_table(report)}

### Case 级 SHA

下表依次为 sparse manifest、Proposed model、fit-first T1、final-pose metrics manifest、common-native metrics manifest。

{_case_table(report)}

### 每个 case 的精确现场文件位置与 SHA

{_case_artifact_details(report)}

新归档尚未开始建立。它完成后会在归档根生成第二套中英文 README，使用通用 launcher 与归档内路径；本 README 永远保留为当前分散现场路线的证据。
"""


def _en_readme(report: dict[str, Any], semantic: dict[str, Any]) -> str:
    command = _commands(report)
    subject = command["subject"]
    roots = report["roots"]
    protocol = semantic["protocol"]
    approved_cases = [case for case, value in semantic["cases"].items() if value.get("status") == "PASS"]
    return f"""# SCMR-v5 MultiMap Frozen Live-Route Record (English)

This is a **beginner-operable manual for the already working live route**. Follow the operational steps first; use the appendix only when exact `{subject}` paths, SHA256 values, and frozen versions are needed. This records successful files where they currently live. Never move, rename, overwrite, or edit the original inputs or outputs.

Software, configurations, and formal launchers must remain subject-agnostic. `{subject}` is only the example dataset in this freeze record.

## Step-by-step operation

### 0. Choose the correct execution location and protect the live route

### Where each stage runs

| Work | Location | Input | Output | Purpose |
|---|---|---|---|---|
| Inspect code, versions, and SHA | local or server code checkout | freeze evidence and original files | terminal checks | prevents mixing routes. |
| SCMR-v5 MATLAB fitting | server with MATLAB | `preprocessed.mat` | mapping and native cache for all stacks | creates correct native T1/T2/B1. |
| 2D-fit-first / NeSVoR | server GPU | native cache and sparse input | T1/T2 volumes | reconstructs the fit-first comparator. |
| Proposed / MLP | server GPU | sparse input, frozen decoder/B6/protocol | checkpoint and T1/T2 volumes | reconstructs the Proposed method. |
| final pose and metrics | server | both methods' volumes and ROI | metrics, AHA, QC figures | compares both methods in one pose/ROI. |

Any command that creates output must use a **new empty run root**. Do not rerun reconstruction in the completed directories recorded in the appendix. Evaluation can read completed reconstructions.

### 1. Set server variables and perform the minimum preflight

Input: server code checkout, prepared observations, ROI, and the two method roots. Output: terminal-only checks; no data are written.

```bash
CODE="{command['code']}"
SUBJECT="{subject}"
PROPOSED_RUN="{command['proposed_run']}"
FIT_RUN="{command['fit_run']}"
PREPARED_ROOT="{command['prepared_root']}"
ROI_BUNDLE="{command['roi']}"

test -d "$CODE" && test -d "$PREPARED_ROOT/$SUBJECT" && test -d "$ROI_BUNDLE"
git -C "$CODE" rev-parse HEAD
```

Pass condition: the directories exist and the Git commit equals the appendix. If it does not, stop; do not infer provenance from a directory name.

### 2. Verify shared inputs and the sparse selection

Input: `$PREPARED_ROOT/$SUBJECT/{{sax,2ch,4ch}}/observations.npz` and each case's `sparse_manifest.json`. Output: observation SHA256 values and the sparse-manifest list.

```bash
for STACK in sax 2ch 4ch; do
  sha256sum "$PREPARED_ROOT/$SUBJECT/$STACK/observations.npz"
done

find "$PROPOSED_RUN/sparse_inputs/$SUBJECT" -name sparse_manifest.json -print
```

Purpose: prepared observations are the common 10-weight raw signal for both methods. `sparse_manifest.json` is authoritative for retained and held-out groups. Pass condition: the three SHA256 values and case names agree with the appendix. Do not use historical `ismrm_v1` inputs or a generic/TRAD cache.

### 3. Run correct SCMR-v5 mapping and fit-first reconstruction on the server

Input: the three `preprocessed.mat` files, prepared observations, and sparse inputs copied into a new root. Output: `mapping/scmr_v5/$SUBJECT/*_mapping.mat`, `native_cache/${{SUBJECT}}_scmr_v5`, and `reconstruction/fit_first/$SUBJECT/<CASE>/`.

Create a new root and copy sparse inputs (copy only; never move existing files):

```bash
NEW_FIT_RUN="/path/to/new_run"
mkdir -p "$NEW_FIT_RUN"
cp -a "$PROPOSED_RUN/sparse_inputs" "$NEW_FIT_RUN/"

bash "$CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_{command['subject_script']}_scmr_v5_fit_first.sh" \\
  "$NEW_FIT_RUN" \\
  "{command['preprocessed_root']}" \\
  "$PREPARED_ROOT" \\
  matlab
```

Purpose: the script runs the SCMR-v5 MATLAB matcher, packages the native cache, then runs fit-first/NeSVoR for sparse cases. Pass condition: each stack `manifest.json` contains `source_observations.sha256`, `source_preprocessed.sha256`, `scmr_v5_mapping.sha256`, and `native_maps_sha256`; each case has `input_manifest.json`, `commands.sh`, and T1/T2 `volume_ms.nii.gz`. The required contract is `TI=[10,100] ms`, `TR={protocol['tr_ms']} ms`, and `VPS={protocol['vps']}`, not the historical `TI=[50,150]` route.

### 4. Run or verify Proposed on the server

Input: the case sparse input, frozen decoder/B6, correct protocol, and `config_resolved.yaml`. Output: `model.pt`, `T1_3D.nii.gz`, `T2_3D.nii.gz`, `experiment_manifest.json`, and `run_provenance.json`.

Before starting, take decoder, B6, protocol, and expected SHA256 values from the **same case's** `config_resolved.yaml`, `experiment_manifest.json`, and `run_provenance.json`; never accept a script default protocol. After a run, check:

```bash
CASE="L100_fold0"  # replace with the target case
find "$PROPOSED_RUN/reconstruction/proposed/$SUBJECT/$CASE" -maxdepth 1 \\
  \\( -name model.pt -o -name T1_3D.nii.gz -o -name T2_3D.nii.gz -o \\
     -name config_resolved.yaml -o -name experiment_manifest.json -o -name run_provenance.json \\) -print
```

Pass condition: all six artifact types are present and the case SHA256 values in the appendix and semantic verification are PASS. Proposed consumes raw signal; it does not take an erroneous fitting cache as its input.

### 5. Run final-pose, AHA, and myocardial/LV evaluation on the server

Input: completed Proposed root, completed fit-first root, and the same ROI bundle. Output: final-pose bundles, AHA (legacy and 1px), myocardial/LV metrics, common-native all-stack/held-out/blood-pool/zero-audit reports, and QC figures.

```bash
CASE="L100_fold0"  # one or more completed cases
bash "$CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/{command['subject_script']}_paired_final_pose.sh" \\
  "$PROPOSED_RUN" "$FIT_RUN" "$ROI_BUNDLE" "$CASE"

bash "$CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/{command['subject_script']}_common_native_metrics_qc.sh" \\
  "$PROPOSED_RUN" "$FIT_RUN" "$ROI_BUNDLE" "$CASE"

OUTPUT_NAME="all_stack_roi_metrics_v2_blood_pool_zero_audit" \\
bash "$CODE/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/{command['subject_script']}_common_native_all_stack_roi_metrics.sh" \\
  "$PROPOSED_RUN" "$FIT_RUN" "$ROI_BUNDLE" "$CASE"
```

Purpose: final pose evaluates both methods on common native query planes. Myocardial reporting retains `myocardium_core_1px` and `myocardium_core_legacy`; blood pool is always `inner_filled AND outer_filled`; LV is `outer_filled`. Pass condition: manifests record final pose, masks, ROI voxel count, all-slice/held-out strata, and the zero audit. Fit-first zeros are not automatically excluded; inspect the zero-audit values.

### 6. Perform the reproducibility check

Input: newly generated manifests and the freeze directory. Output: a SHA comparison record.

```bash
sha256sum <file>
grep -F '"path": "<original-path>"' "<freeze_root>/source_file_manifest.jsonl"
```

Only declare a copied route reproducible when source and copied SHA256 values agree, all Proposed cases pass semantic verification, and all native-cache provenance values agree. Never use a reconstructed 3D volume as the AHA native 2D reference, and do not transpose SAX ROI (native orientation is `[group,row,col]`).

## Frozen live files and SHA256 appendix

### Frozen state

- Frozen code commit: `{report['code']['git_head']}`
- Code root: `{report['code']['path']}`
- Git status: `code/git_status.txt`
- Tracked working-tree patch: `code/git_worktree.patch`
- Git bundle: `code/code_head.bundle`
- Example subject: `{subject}`
- Cases passing semantic verification: {', '.join(approved_cases)}
- Correct protocol: `{protocol['path']}` (SHA256: `{protocol['sha256']}`)
- Fitting timing: `delta_AcquisitionTime - TR*(EchoTrainLength+10) - preparation_delay`; normalized absolute 10D matching, QC 0.90, no interpolation/refinement.

### Live paths

| Category | Path | Purpose |
|---|---|---|
| prepared observations | `{roots['shared_prepared_observations']}` | Prepared observations for all stacks. |
| manual ROI | `{roots['shared_manual_roi']}` | outer/inner filled, blood pool, myocardial ROI. |
| sparse inputs | `{roots['shared_sparse_inputs']}` | sparse manifest and retained observations per case. |
| MATLAB mapping | `{roots['fit_first_matlab_mapping']}` | corrected mapping. |
| SCMR-v5 native cache | `{roots['fit_first_native_cache']}` | authoritative fit-first/native-reference cache. |
| fit-first outputs | `{roots['fit_first_reconstructions']}` | NeSVoR outputs. |
| Proposed outputs | `{roots['proposed_reconstructions']}` | MLP outputs and provenance. |
| final-pose evaluation | `{roots['final_pose_evaluation']}` | AHA, myocardial/LV metrics, QC. |
| common-native evaluation | `{roots['common_native_evaluation']}` | all-stack, held-out, blood-pool, zero audit. |

### SCMR-v5 native-cache SHA256

{_native_table(report)}

### Per-case SHA256

Columns are sparse manifest, Proposed model, fit-first T1, final-pose metrics manifest, and common-native metrics manifest.

{_case_table(report)}

### Exact live file locations and SHA256 for every case

{_case_artifact_details(report)}

The new archive has not been created yet. When it is ready, a second bilingual README will be generated in the archive root using generic launchers and archive-local paths. This README remains the evidence for the scattered live route.
"""


def build_freeze_readmes(freeze_root: str | Path) -> tuple[Path, Path]:
    """Write Chinese and English live-route records from verified freeze evidence."""
    freeze = Path(freeze_root)
    report = _load(freeze / "live_route_freeze.json")
    semantic = _load(freeze / "proposed_semantic_verification.json")
    if any(value.get("status") != "PASS" for value in semantic.get("cases", {}).values()):
        raise ValueError("refusing README generation: Proposed semantic verification is not fully PASS")
    zh, en = freeze / "README_zh-CN.md", freeze / "README_en.md"
    zh.write_text(_zh_readme(report, semantic), encoding="utf-8")
    en.write_text(_en_readme(report, semantic), encoding="utf-8")
    return zh, en


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-root", required=True)
    for path in build_freeze_readmes(parser.parse_args().freeze_root):
        print(path)


if __name__ == "__main__":
    main()
