# ismrm_v2_1 独立 SCMR-v5 MultiMap Pipeline 设计

## Intent

建立一个独立、可复现、可继续运行的 `ismrm_v2_1` 项目，作为当前 corrected SCMR-v5 MultiMap pipeline 的唯一后续运行入口。项目必须能够从原始 DICOM 开始，完成服务器端预处理、prepared observations、SCMR-v5 MATLAB fitting、2D-fit-first、Proposed、final-pose、AHA、左心室/心肌/血池 metrics 和 QC 可视化；后续运行不再依赖 `/data/dengyz/code/Code_10w`。

当前四个 subject 的运行必须继续在现有运行根目录中完成；新项目只进行只读核对和复制，不移动或覆盖正在运行的文件。

## Goals

- 创建 `/data/dengyz/code/ismrm_v2_1` 作为代码和项目数据的唯一根目录。
- 只复制当前 corrected pipeline 实际使用的代码闭包，不复制早期错误路线、无关 synthetic 实验、废弃 CYJ 专用入口和 `_garbage`。
- 用通用 `SUBJECT`、case 和 manifest 参数支持 HHZ、CYJ、DYZ、DYL_2、HJL，不在核心代码中写死 subject。
- 把协议、模型、运行配置和入口脚本集中到 `configs/`、`runtime/`、`scripts/`，并记录 SHA256。
- 在项目内保存五个 subject 的输入/输出目录结构；原始数据和大结果不进入 GitHub，但保留 manifest、provenance 和 SHA 记录。
- 提供中英文 README，按真实 pipeline 顺序说明服务器执行位置、输入、输出、作用、命令、失败排查和完整性核对。
- 为新目录提供一次独立 preflight，确认不再解析 `Code_10w` 的代码、旧 decoder 或旧 protocol。

## Non-goals

- 不修改或中断当前四个 subject 的旧运行。
- 不把原始 DICOM、病人 ROI、模型大文件或完整重建结果提交到 GitHub。
- 不重写已经验证过的 MATLAB matcher、NeSVoR、Proposed、metrics 算法；只做路径参数化和必要的运行入口适配。
- 不把历史 `[50,150]` decoder 作为当前 corrected pipeline 的可用依赖。
- 不把 HHZ、CYJ 等 subject 名称写入通用核心模块；subject-specific 信息只进入 manifest、数据目录和运行参数。

## Selected approach

采用“精简独立仓库”方案：从当前实际运行路线的入口和 import 闭包出发复制代码，建立新的 Git 仓库；输入和输出作为项目外观上的子目录存在，但由 `.gitignore` 排除。与薄包装方案相比，该方案不再依赖 `Code_10w` 的绝对路径；与完整复制后删减相比，该方案不会把历史错误入口带入新运行环境。

## Repository and data layout

```text
ismrm_v2_1/
├── configs/
│   ├── protocol_scmr_v5_10_100.yaml
│   ├── subject_stack_manifest.example.json
│   ├── subject_stack_manifest.json
│   └── models_manifest.json
├── pipeline/
│   ├── 01_RAW_DICOM/
│   │   ├── scripts/
│   │   ├── README_zh-CN.md
│   │   └── README_en.md
│   ├── 02_PREPROCESS_MATLAB/
│   │   ├── matlab/
│   │   ├── scripts/
│   │   ├── README_zh-CN.md
│   │   └── README_en.md
│   ├── 03_PREPARE_OBSERVATIONS/
│   │   ├── python/
│   │   ├── scripts/
│   │   ├── README_zh-CN.md
│   │   └── README_en.md
│   ├── 04_SPARSE_PROFILES/
│   │   ├── python/
│   │   ├── scripts/
│   │   ├── README_zh-CN.md
│   │   └── README_en.md
│   ├── 05_SCMR_V5_FIT_FIRST/
│   │   ├── matlab/
│   │   ├── python/
│   │   ├── scripts/
│   │   ├── README_zh-CN.md
│   │   └── README_en.md
│   ├── 06_PROPOSED/
│   │   ├── python/
│   │   ├── scripts/
│   │   ├── README_zh-CN.md
│   │   └── README_en.md
│   ├── 07_FINAL_POSE/
│   │   ├── python/
│   │   ├── scripts/
│   │   ├── README_zh-CN.md
│   │   └── README_en.md
│   ├── 08_EVALUATION_AND_VISUALIZATION/
│   │   ├── python/
│   │   ├── scripts/
│   │   ├── README_zh-CN.md
│   │   └── README_en.md
│   └── 09_INTEGRITY_AND_REPORTING/
│       ├── python/
│       ├── scripts/
│       ├── README_zh-CN.md
│       └── README_en.md
├── common/
│   ├── trad/
│   ├── mlp/
│   ├── reconstruction_core/
│   ├── quality_experiments/
│   └── paper_experiments/
├── scripts/
│   ├── run_pipeline.sh
│   ├── verify_project_integrity.py
│   └── verify_runtime_contract.py
├── runtime/
│   ├── model_assets/
│   ├── environment/
│   └── source_records/
├── subjects/
│   ├── HHZ/
│   ├── CYJ/
│   ├── DYZ/
│   ├── DYL_2/
│   └── HJL/
├── runs/
│   ├── HHZ/
│   ├── CYJ/
│   ├── DYZ/
│   ├── DYL_2/
│   └── HJL/
├── docs/
│   ├── README_zh-CN.md
│   ├── README_en.md
│   ├── pipeline_contract.md
│   └── migration_record.md
├── tests/
├── .gitignore
└── VERSION
```

`subjects/<SUBJECT>/` contains subject-scoped inputs and immutable source identities: raw DICOM reference, deployment manifest, `preprocessed.mat`, preprocessing QC, prepared observations, ROI bundle and sparse manifests. `runs/<SUBJECT>/` contains fitting, reconstruction, final-pose, metrics and visualization outputs. Current live runs are first copied into these locations after completion; existing live paths remain authoritative until their copy is verified.

Every pipeline module is prefixed with an Arabic-numeral order (`01_` through `09_`) and contains two operator-facing documents: `README_zh-CN.md` and `README_en.md`. Each module README explains the module purpose, the exact file execution order, every code file's responsibility, inputs, outputs, server/local execution location, command examples, required SHA/manifest checks, expected success markers and troubleshooting. `common/` retains import-compatible package names; the numbered module directories are the operator-facing organization and entrypoint layer.

## Runtime contracts

The current corrected route uses:

- protocol: SCMR-v5, FA `[45,45,45]`, TI `[10,100]`, T2-prep `[35,45,55]`, ramp-up `10`, TR `2.61 ms`, VPS `87`;
- decoder: the approved `[10,100]` checkpoint at `/data/dengyz/mlp_candidate_hhz_scmr_v5/signal_simulator_approved.pth`, SHA256 `338f2c0b8db90851f733ed31caa2c99914605fed3f7215ec2f69f19eba59583b`;
- B6 configuration model: SHA256 `15981fbe93d2c70e14755ea756a02856c9383481e7d92d239619268a628a29b5`;
- corrected protocol YAML: SHA256 `3865b6bde1154750d4ed687456823fafad7b0cad5a61f763decb27dd1a4ed1c7`;
- fit-first and native cache: SCMR-v5 MATLAB dictionary matching with no interpolation/refinement;
- evaluation: final-pose outputs, all-stack sampling where available, LV outer-filled mask, myocardial legacy and 1px cores, blood-pool legacy inner-filled mask, all-slice/held-out/stack strata, zero-value audit and complete metric set.

The old `[50,150]` decoder and its SHA are historical evidence only and must not be copied into `runtime/model_assets/` as an active dependency.

## Pipeline boundaries

1. **Raw DICOM and deployment manifest** — server-side source data. Three stacks (`sax`, `2ch`, `4ch`) and ten weights per group are required.
2. **MATLAB preprocessing** — server-side. Produces `preprocessed.mat` and `preprocess_qc.json`; DICOM geometry, UID and acquisition timing are preserved.
3. **Observation bridge** — server-side Python. Consumes preprocessed MAT plus the explicit corrected protocol and produces `observations.npz`, `manifest.json`, timing and QC summaries.
4. **Sparse profiles** — server-side Python. Produces immutable sparse manifests and cropped observation sets; L100 points to the subject prepared root, while profile cases point to their derived case root.
5. **SCMR-v5 native cache and fit-first** — server MATLAB plus NeSVoR. Produces native maps, native cache manifests, 2D-fit-first reconstructions and provenance bindings.
6. **Proposed** — server CUDA. Consumes sparse observations, the explicit corrected decoder, B6 model, protocol and SHA pins; produces maps, signals, model checkpoints and provenance.
7. **Final pose and evaluation** — server Python. Produces final-pose bundles, AHA maps/bullseyes, paired slice/through-plane QC, LV/myocardial/blood-pool metrics and summary tables.

Every boundary records absolute paths, SHA256, code commit, protocol record, subject, case and command. A downstream step must refuse a missing or mismatched identity instead of silently substituting a historical file.

## Migration policy

- Copy, never move, current live files.
- Copy source code from the verified corrected commit after recording the actual server `HEAD` used by the live runs.
- Copy the correct decoder into `runtime/model_assets/decoder_scmr_v5_10_100.pth`; do not copy the old archive decoder.
- Copy the corrected protocol into `configs/protocol_scmr_v5_10_100.yaml` and record its SHA.
- Copy HHZ as the verified reference instance and copy CYJ, DYZ, DYL_2 and HJL only after their live runs reach a documented terminal state.
- Create a migration manifest for every copied file with source path, destination path, size, SHA256 and copy timestamp.
- Keep a separate record of files intentionally excluded: old routes, `_garbage`, synthetic experiments, old result trees and legacy subject-only wrappers.

## Acceptance criteria

- `verify_project_integrity.py` reports all tracked source files and active runtime assets present, with no active dependency on `/data/dengyz/code/Code_10w`.
- `verify_runtime_contract.py` reports protocol, decoder, B6 and manifest SHA matches, and rejects the old `[50,150]` decoder.
- A dry-run command for each subject resolves all input/output roots under `ismrm_v2_1`.
- A clean test checkout can run the command builders and manifest validators without patient data.
- README commands are sufficient for a new operator to run the full pipeline from DICOM through metrics without consulting the historical `Code_10w` tree.

## Open implementation decisions

- The exact corrected server `HEAD` and dirty-tree patch used by the four live runs must be recorded before copying the code closure.
- The final physical location of the five subject data copies will be decided during migration; no live source is moved.
- The GitHub remote URL and repository visibility are external choices and are not required for the local reproducibility package.
