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


def _zh_readme(report: dict[str, Any], semantic: dict[str, Any]) -> str:
    subject = report["subject"]
    roots = report["roots"]
    protocol = semantic["protocol"]
    approved_cases = [case for case, value in semantic["cases"].items() if value.get("status") == "PASS"]
    return f"""# SCMR-v5 MultiMap 已冻结现场路线记录（中文）

## 文档目的与边界

本文件记录当前已经跑通、但仍分布在原服务器路径中的路线。它是**现场冻结记录**，不是移动后的新归档运行手册。原始文件不得移动、重命名、覆盖或修改；其完整文件清单及 SHA256 位于 `source_file_manifest.jsonl`，机器可读路线证据位于 `live_route_freeze.json`。

本例 subject 为 `{subject}`。软件逻辑、config 和将来的正式 launcher 必须对任意 subject 通用；仅数据与结果目录使用 subject ID。

## 当前冻结状态

- 冻结代码 Git commit：`{report['code']['git_head']}`
- 代码根目录：`{report['code']['path']}`
- Git 状态：`code/git_status.txt`
- 已跟踪改动 patch：`code/git_worktree.patch`
- Git bundle：`code/code_head.bundle`
- 当前 subject：`{subject}`
- Proposed 语义核验通过的 case：{', '.join(approved_cases)}

## 正确 scientific contract（不得替换为旧路线）

- Protocol 文件：`{protocol['path']}`
- Protocol SHA256：`{protocol['sha256']}`
- fitting inversion times：`TI=[10,100] ms`
- `TR={protocol['tr_ms']} ms`，`VPS={protocol['vps']}`
- timing：`delta_AcquisitionTime - TR*(EchoTrainLength+10) - preparation_delay`
- fitting：SCMR-v5 MATLAB dictionary matching，normalized absolute 10D inner product，QC threshold 0.90，无 interpolation/refinement。
- 不可使用旧 `TI=[50,150]` fitting route、旧 generic/TRAD native cache 或旧 fit-first 输出作为本路线输入。

## 在哪里运行

| 阶段 | 地点 | 作用 |
|---|---|---|
| 代码检查、编辑、审计设计 | 本地或服务器 code checkout | 不产生正式重建结果。 |
| MATLAB SCMR-v5 fitting | 服务器 | 生成三 stack native T1/T2/B1 mapping。 |
| Fit-first / NeSVoR | 服务器 GPU | 从 native map 重建 T1/T2 volume。 |
| Proposed / MLP | 服务器 GPU | 使用冻结 decoder、B6、sparse input 重建。 |
| final-pose、AHA、ROI、common-native | 服务器 | 生成所有定量表和 QC 图。 |

## 当前现场路径：输入、输出与作用

| 类别 | 路径 | 作用 |
|---|---|---|
| prepared observations | `{roots['shared_prepared_observations']}` | 三个 stack 的已准备 10-weight observations。 |
| manual ROI | `{roots['shared_manual_roi']}` | outer/inner filled、blood pool、三种 myocardium ROI。 |
| sparse inputs | `{roots['shared_sparse_inputs']}` | 每个 case 的 sparse manifest 与 retained observations。 |
| MATLAB mapping | `{roots['fit_first_matlab_mapping']}` | 修正参数后的 `*_mapping.mat`。 |
| SCMR-v5 native cache | `{roots['fit_first_native_cache']}` | Fit-first 与 native-reference 的权威 cache。 |
| fit-first outputs | `{roots['fit_first_reconstructions']}` | NeSVoR T1/T2 volumes、input manifest、commands。 |
| Proposed outputs | `{roots['proposed_reconstructions']}` | Proposed checkpoints、T1/T2、resolved config、provenance。 |
| final-pose evaluation | `{roots['final_pose_evaluation']}` | AHA、心肌/LV metrics、pair-slice、through-plane QC。 |
| common-native evaluation | `{roots['common_native_evaluation']}` | all-stack、held-out、blood-pool、zero audit。 |

## SCMR-v5 native-cache 核对

{_native_table(report)}

运行任何下游步骤前，必须从对应 `manifest.json` 核对 `source_observations.sha256`、`source_preprocessed.sha256`、`scmr_v5_mapping.sha256` 与 `native_maps_sha256`。不得仅通过目录名称判断正确性。

## Case 级输出 SHA 核对

下表中的 SHA 依次为 sparse manifest、Proposed model、fit-first T1、final-pose metrics manifest、common-native metrics manifest。

{_case_table(report)}

每个 Proposed case 同时必须存在并与冻结记录一致：`model.pt`、`T1_3D.nii.gz`、`T2_3D.nii.gz`、`config_resolved.yaml`、`experiment_manifest.json`、`run_provenance.json`。

## 每个 case 的精确现场文件位置与 SHA

{_case_artifact_details(report)}

## 现场路线的正确运行顺序

1. 以当前 cache manifest 中的 `source_preprocessed.path` 为准，准备三个 `preprocessed.mat`。
2. 生成或核对 `prepared observations`；每个 spatial group 必须有 weight `0..9`，TR=2.61、VPS=87。
3. 生成 sparse inputs；每个 case 的 `sparse_manifest.json` 是 retained/held-out group 与 generator Git SHA 的权威记录。
4. 在服务器 MATLAB 中运行 SCMR-v5 matcher，生成 `sax/2ch/4ch_mapping.mat`。
5. 打包 immutable native cache，并核对三个 stack 的 source SHA。
6. 使用该 cache 执行 fit-first + NeSVoR；保存 `input_manifest.json`、`commands.sh` 与两个 `volume_ms.nii.gz`。
7. 使用 Proposed case 的 `config_resolved.yaml`、`experiment_manifest.json`、`run_provenance.json` 中记录的 decoder、B6、protocol 与 SHA 执行 Proposed。禁止依赖旧 protocol 默认值。
8. 在服务器执行 final-pose/AHA/ROI/common-native evaluation；指标使用各输出 manifest 绑定的输入，不重跑 reconstruction。

## 当前现场下游命令模式

当前 live route 的下游脚本接受两个不同根：Proposed 根与修复后 fit-first 根。调用时必须分别传入，不能都传同一个目录。

```bash
bash <live-route-paired-final-pose-launcher>.sh \\
  "$PROPOSED_RUN" "$FIT_RUN" "$ROI_BUNDLE" <CASE...>
```

这里 `$PROPOSED_RUN` 是 Proposed outputs 所在运行根，`$FIT_RUN` 是包含 SCMR-v5 native cache 和 fit-first outputs 的运行根。该脚本为当前现场证据，未来归档将替换为不写死 subject 的 launcher。

## 禁止混用

- 不能把 `ismrm_v1` 当作本路线输入；它是旧 protocol 历史路线。
- 不能把旧 generic/TRAD `native_cache/subject` 当作 SCMR-v5 native cache。
- 不能转置 SAX ROI；native mapping 与 ROI 均保持 `[group,row,col]` 原始方向。
- 不能用 3D reconstructed volume 替代 AHA native 2D reference。
- blood pool 使用 `inner_filled AND outer_filled`，不使用 1-pixel myocardium core。
- fit-first 数值 0 不会自动排除；应检查 all-stack report 的 zero-audit 字段。

## SHA 核对命令

```bash
sha256sum <file>
grep -F '"path": "<original-path>"' source_file_manifest.jsonl
```

正式复制到新归档后，必须比较原文件与复制文件的 SHA256；一致后才能将复制品声明为封存副本。

## 后续文档

新归档建立完成后，将在归档根生成另一套中英文 README。后者使用通用 launcher、归档内相对路径和可复跑步骤；本文件始终保留当前分散现场路线的历史证据。
"""


def _en_readme(report: dict[str, Any], semantic: dict[str, Any]) -> str:
    subject = report["subject"]
    roots = report["roots"]
    protocol = semantic["protocol"]
    approved_cases = [case for case, value in semantic["cases"].items() if value.get("status") == "PASS"]
    return f"""# SCMR-v5 MultiMap Frozen Live-Route Record (English)

## Purpose and scope

This document records the currently working route while its artifacts still live in their original server locations. It is a **live-route freeze record**, not the future relocated archive manual. Do not move, rename, overwrite, or edit original artifacts. The complete file-level SHA256 inventory is `source_file_manifest.jsonl`; the machine-readable route evidence is `live_route_freeze.json`.

The example subject is `{subject}`. Software, configurations, and future launchers must remain subject-agnostic; only data and result directories contain subject IDs.

## Frozen state

- Frozen code commit: `{report['code']['git_head']}`
- Code root: `{report['code']['path']}`
- Git status: `code/git_status.txt`
- Tracked working-tree patch: `code/git_worktree.patch`
- Git bundle: `code/code_head.bundle`
- Example subject: `{subject}`
- Proposed cases passing semantic verification: {', '.join(approved_cases)}

## Correct scientific contract

- Protocol file: `{protocol['path']}`
- Protocol SHA256: `{protocol['sha256']}`
- Fitting inversion times: `TI=[10,100] ms`
- `TR={protocol['tr_ms']} ms`, `VPS={protocol['vps']}`
- Timing: `delta_AcquisitionTime - TR*(EchoTrainLength+10) - preparation_delay`
- Fitting: SCMR-v5 MATLAB dictionary matching, normalized absolute 10D inner product, QC threshold 0.90, no interpolation/refinement.
- Do not substitute the historical `TI=[50,150]` route, a generic/TRAD native cache, or old fit-first outputs.

## Where each stage runs

| Stage | Location | Purpose |
|---|---|---|
| Code inspection, editing, and audit design | local or server checkout | Does not create formal reconstruction results. |
| MATLAB SCMR-v5 fitting | server | Produces native T1/T2/B1 maps for all three stacks. |
| Fit-first / NeSVoR | server GPU | Reconstructs T1/T2 volumes from native maps. |
| Proposed / MLP | server GPU | Reconstructs from frozen decoder, B6, and sparse inputs. |
| final-pose, AHA, ROI, common-native evaluation | server | Produces quantitative tables and QC figures. |

## Live paths: inputs, outputs, and purpose

| Category | Path | Purpose |
|---|---|---|
| prepared observations | `{roots['shared_prepared_observations']}` | Prepared 10-weight observations for three stacks. |
| manual ROI | `{roots['shared_manual_roi']}` | outer/inner filled, blood pool, and three myocardial ROIs. |
| sparse inputs | `{roots['shared_sparse_inputs']}` | Sparse manifests and retained observations per case. |
| MATLAB mapping | `{roots['fit_first_matlab_mapping']}` | Corrected `*_mapping.mat` files. |
| SCMR-v5 native cache | `{roots['fit_first_native_cache']}` | Authoritative cache for fit-first and native reference. |
| fit-first outputs | `{roots['fit_first_reconstructions']}` | NeSVoR volumes, input manifest, and commands. |
| Proposed outputs | `{roots['proposed_reconstructions']}` | Checkpoints, T1/T2 volumes, resolved config, provenance. |
| final-pose evaluation | `{roots['final_pose_evaluation']}` | AHA, myocardial/LV metrics, pair-slice, through-plane QC. |
| common-native evaluation | `{roots['common_native_evaluation']}` | all-stack, held-out, blood-pool, and zero-value audit. |

## SCMR-v5 native-cache checks

{_native_table(report)}

Before any downstream step, verify `source_observations.sha256`, `source_preprocessed.sha256`, `scmr_v5_mapping.sha256`, and `native_maps_sha256` in the matching `manifest.json`. Directory names alone are not provenance.

## Per-case SHA checks

The following SHA256 columns are: sparse manifest, Proposed model, fit-first T1, final-pose metrics manifest, and common-native metrics manifest.

{_case_table(report)}

Each Proposed case must also retain `model.pt`, `T1_3D.nii.gz`, `T2_3D.nii.gz`, `config_resolved.yaml`, `experiment_manifest.json`, and `run_provenance.json`.

## Exact live file locations and SHA256 for every case

{_case_artifact_details(report)}

## Correct live-route order

1. Use the native-cache manifest's `source_preprocessed.path` to identify the three `preprocessed.mat` inputs.
2. Build or verify prepared observations: every spatial group has weights `0..9`, with TR=2.61 and VPS=87.
3. Build sparse inputs; each `sparse_manifest.json` is authoritative for retained/held-out groups and generator Git SHA.
4. Run the SCMR-v5 MATLAB matcher on the server and produce `sax/2ch/4ch_mapping.mat`.
5. Package the immutable native cache and verify source SHA256 values for all stacks.
6. Run fit-first + NeSVoR, retaining `input_manifest.json`, `commands.sh`, and both `volume_ms.nii.gz` files.
7. Run Proposed using the decoder, B6, protocol, and SHA256 values recorded in that case's resolved configuration and provenance. Never rely on an old default protocol.
8. Run final-pose/AHA/ROI/common-native evaluation on the server without rerunning reconstruction.

## Current downstream command pattern

The live route takes two different roots: a Proposed root and a corrected fit-first root. They must not be passed as the same directory.

```bash
bash <live-route-paired-final-pose-launcher>.sh \\
  "$PROPOSED_RUN" "$FIT_RUN" "$ROI_BUNDLE" <CASE...>
```

This subject-specific script is retained as evidence for the live route. The future archive will provide a subject-agnostic launcher.

## Do not mix

- Do not use `ismrm_v1` as an input to this route.
- Do not use an old generic/TRAD cache as the SCMR-v5 native cache.
- Do not transpose the SAX ROI; maps and ROI retain their native `[group,row,col]` orientation.
- Do not substitute a reconstructed 3D volume for the AHA native 2D reference.
- Blood pool is `inner_filled AND outer_filled`, not a 1-pixel myocardial core.
- Numeric fit-first zeros are not automatically excluded; inspect the all-stack zero-audit fields.

## SHA256 command

```bash
sha256sum <file>
grep -F '"path": "<original-path>"' source_file_manifest.jsonl
```

After copying to the future archive, compare each source and copied SHA256 before declaring the copied artifact frozen.

## Next documentation

Once the new archive is complete, a second bilingual README will be generated at its root. It will use generic launchers, archive-relative paths, and reproducible run steps. This document remains the historical evidence for the original scattered live route.
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
