# ismrm_v2_1

这是 SCMR-v5 MultiMap 的独立运行目录模板。代码按流水线顺序放在 `pipeline/01_` 到 `pipeline/09_`；每个目录同时有中英文操作手册和 `run_stage.sh`。`common/` 保存运行时所需的 import-compatible 算法代码，`configs/` 保存 protocol/manifest 模板，`subjects/` 保存本地 subject 输入副本，`runs/` 保存运行结果。后两类数据默认不进 Git。

## 运行原则

1. 所有新实验只从本目录运行；不要再引用旧的 `Code_10w` 绝对路径。
2. 服务器完成 DICOM 预处理、observations、SCMR-v5 MATLAB fitting、2D-fit-first、Proposed、final pose 和 metrics；本地只负责手工 ROI/mask，然后把 ROI 文件复制到服务器。
3. 任何新 subject 都用显式 `SUBJECT` 和单 subject deployment manifest；核心代码不写死 subject 名。
4. protocol 必须是 TR=2.61、VPS=87、TI=[10,100]、T2prep=[35,45,55]、FA=[45,45,45]、ramp=10。Proposed 只使用 SHA 为 `338f2c...` 的正确 decoder；旧 `[50,150]` decoder 必须拒绝。
5. 先完成前一模块并检查 manifest/SHA/PASS，再执行后一模块。所有重跑写入新的 `runs/<SUBJECT>/<RUN_ID>`。

## 从 DICOM 开始

```bash
PROJECT_ROOT=/data/dengyz/code/ismrm_v2_1
SUBJECT=HHZ
RUN_ROOT=/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/ismrm_v2_1/$SUBJECT/run_$(date +%Y%m%d_%H%M%S)
PYTHON=/home/dengyz/miniconda3/envs/cr_dreme/bin/python
mkdir -p "$RUN_ROOT"
```

依次阅读并运行：

```bash
"$PROJECT_ROOT/pipeline/01_RAW_DICOM/run_stage.sh" "$SUBJECT" /data/dengyz/dataset/mapping_subject_dicom
"$PROJECT_ROOT/pipeline/02_PREPROCESS_MATLAB/run_stage.sh" "$RUN_ROOT" "$SUBJECT" "$PROJECT_ROOT/subjects/$SUBJECT/manifest.json" matlab "$PYTHON"
"$PROJECT_ROOT/pipeline/03_PREPARE_OBSERVATIONS/run_stage.sh" "$RUN_ROOT" "$SUBJECT" "$PROJECT_ROOT/subjects/$SUBJECT/manifest.json" "$PYTHON"
"$PROJECT_ROOT/pipeline/04_SPARSE_PROFILES/run_stage.sh" "$RUN_ROOT" "$SUBJECT" "$PYTHON"
"$PROJECT_ROOT/pipeline/05_SCMR_V5_FIT_FIRST/run_stage.sh" "$RUN_ROOT" "$SUBJECT" matlab
```

然后对八个 case 分别运行模块 06；模块 07/08 对每个已完成 case 运行。每个模块的输入、输出、服务器位置、SHA 检查和故障排除在对应 README 中给出。

## 快速验证

```bash
SUBJECT=HHZ DRY_RUN=1 "$PROJECT_ROOT/scripts/run_pipeline.sh"
python "$PROJECT_ROOT/scripts/verify_project_integrity.py" "$PROJECT_ROOT"
python "$PROJECT_ROOT/scripts/verify_no_legacy_code_path.py" "$PROJECT_ROOT"
```

不要把原始 DICOM、MAT/NPZ/NIfTI、模型二进制或结果树提交到 GitHub；提交 manifests、SHA、配置、脚本和 README。服务器同步后，应先运行 09 模块，再开始新 subject。
