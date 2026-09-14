# 双环境部署流程（v1）

此部署层不改变已冻结的 HHZ v1 科学模型、Trad/MLP INR、rigid/PSF 或训练目标；不包含 EPG、deformable 或 low-rank。

## 明确清单与本地 MATLAB

将 `configs/subject_stack_manifest.example.json` 复制到仓库外，并为每个 subject/stack 写入精确 `dicom_dir`。脚本绝不扫描目录推断 subject 或 stack。先创建每个 `<preprocessed_root>/<subject_id>/<stack>/`，再在本地 MATLAB 中运行：

```matlab
addpath('/absolute/path/to/Code_10w/trad/scripts')
local_preprocess_all('/absolute/path/subject_stack_manifest.json', ...
                     '/absolute/path/local_preprocessed')
```

每个条目只写 `preprocessed.mat` 和 `preprocess_qc.json`，已有输出会报错。将整个 `local_preprocessed` 传输到服务器；服务器不需要 MATLAB。

## 服务器检查与 observation 准备

```bash
cd /absolute/path/to/Code_10w
conda run -n cr_dreme python scripts/check_server_environment.py --mode formal-mlp
conda run -n cr_dreme python scripts/check_server_environment.py --mode reconstruction

cd trad
conda run -n cr_dreme python scripts/validate_transferred_preprocessed.py \
  --manifest /absolute/path/subject_stack_manifest.json \
  --preprocessed-root /absolute/path/transferred_preprocessed \
  --output /absolute/path/transfer_qc.json
conda run -n cr_dreme python scripts/prepare_all_observations.py \
  --manifest /absolute/path/subject_stack_manifest.json \
  --preprocessed-root /absolute/path/transferred_preprocessed \
  --prepared-root /absolute/path/server_prepared
```

`dicom_dir` 必须是服务器上的明确原始 DICOM 位置。缺 MAT、metadata 或完整十权重 group 会立即失败。每个输出严格位于 `<prepared_root>/<subject_id>/<stack>/`，含 `observations.npz`、`manifest.json`、`timing.npy`、`qc_summary.json`；根目录另有 `PREPARED_BATCH_QC.json`。

## Formal MLP：到人工审核为止

```bash
cd /absolute/path/to/Code_10w/mlp
conda run -n cr_dreme python scripts/build_formal_manifest_from_prepared.py \
  --deployment-manifest /absolute/path/subject_stack_manifest.json \
  --prepared-root /absolute/path/server_prepared \
  --output /absolute/path/formal_timing_manifest.json
```

之后按 `FORMAL_MLP_VALIDATION_v1.md` 的 CUDA 生成、训练和验证命令进行。独立 benchmark 可读取非 functional 的 formal candidate（`unvalidated` 或 `awaiting_manual_review`），但不构成上线批准。验证报告产生后，审核者必须显式执行：

```bash
conda run -n cr_dreme python scripts/approve_formal_checkpoint.py \
  --checkpoint /absolute/path/formal_candidate.pth \
  --validation-report /absolute/path/formal_validation_report.json \
  --approved-output /absolute/path/approved_formal_checkpoint.pth \
  --review-note '明确记录审核结论'
```

该命令核对 checkpoint/report SHA256，复制不变的 state dict 并只加入审核元数据。没有批准状态的 checkpoint 被 MLP 在线重建严格拒绝。

## 单 subject、三 stack 重建

人工批准后，一次只运行一个明确 subject。包装器固定读取 `sax`、`2ch`、`4ch` 的精确 prepared 路径，不搜索目录：

```bash
cd /absolute/path/to/Code_10w/trad
conda run -n cr_dreme python -m scripts.reconstruct_subject \
  --prepared-root /absolute/path/server_prepared --subject-id SUBJECT_ID \
  --config configs/server_train_example.yaml --output /absolute/path/trad_output/SUBJECT_ID

cd /absolute/path/to/Code_10w/mlp
conda run -n cr_dreme python -m scripts.reconstruct_subject \
  --prepared-root /absolute/path/server_prepared --subject-id SUBJECT_ID \
  --config configs/recon_server_train_example.yaml \
  --mlp-checkpoint /absolute/path/approved_formal_checkpoint.pth \
  --output /absolute/path/mlp_output/SUBJECT_ID
```

现有 `run_server_example.sh` 保留环境变量入口，均从脚本自身定位方法根目录。服务器正式命令使用 `cr_dreme`；本地 smoke 仍使用 `knesvr_torch`。本次只交付部署层，不执行正式 preprocessing、training、benchmark 或 reconstruction。
