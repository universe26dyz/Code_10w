# MultiMap 10w Trad v1

Trad 是 subject-specific 的直接定量 SVR：用一个 NeSVoR-style quantitative INR
输出 `T1/T2/B1/A`，对每个 PSF 样本执行 HHZ 直接可微信号模拟，再以 10 个
MIND 配准 observation 的 raw/preprocessed intensity 做 2D data consistency。
它不是传统的“先对每张 2D 图做 dictionary matching、再将定量图交给 NeSVoR”。

```text
DICOM (10 weights/native slice)
  → MATLAB crop → MIND: weight 2..10 → HB1 → Mag_crop
  → MATLAB-v7.3 bridge: timing + cropped HB1 LPS affine + TR/VPS
  → local [column,row,slice] mm / one group pose per slice
  → physical RAS-mm → centered/scaled NeSVoR training space
  → HashGrid shared latent → T1,T2,B1,A
  → HHZ 10-HB direct bSSFP simulator
  → local anisotropic PSF → group rigid → signal-before-average
  → balanced MSE (Stage A INR; Stage B INR + rigid)
  → physical-RAS T1/T2/B1/A NIfTI + final rigid poses
```

## 固定科学协议与数据约束

- HHZ 是 v1 权威：`NumImg=10`、FA=`[45,45,45]`°、TI=`[50,150]` ms、
  T2prep=`[35,45,55]` ms、`nRampUp=10`。
- DYZ 旧代码出现 TI=`[10,100]`；v1 不继承它。
- TR/VPS 必须来自 preprocessing MAT 已验证的 DICOM
  `RepetitionTime`/`EchoTrainLength`，bridge 写入 `observations.npz`，训练时
  所有 group/stack 必须完全一致。
- MIND 在 MATLAB preprocessing 中发生，最终输入必须为
  `Mag_crop = MIND_mag_reg`；10 个 weights 共用 cropped HB1 geometry。

## 模块

| 模块 | 输入 → 输出 | 关键文件与来源 |
|---|---|---|
| 01 preprocessing | DICOM → v7.3 MAT | `preprocessing_v1/`; HHZ 原码保存于 `hhz_original/` |
| 02 bridge | MAT + 当前 DICOM → NPZ/manifest/timing/TR/VPS | `module_02_data_bridge/prepare_observations.py` |
| 03 dataset | NPZ → local pixels、group metadata、physical RAS bbox | `quantitative_point_dataset.py`; NeSVoR PointDataset 思路 |
| 04 INR | scaled RAS → T1/T2/B1/A | `quantitative_inr.py`; vendored NeSVoR HashGrid/build helpers |
| 05A signal | T1/T2/B1/timing → 10-HB fingerprint | `trad_signal_simulator.py`; HHZ `sim_T1T2_10HB_bssfp.m` |
| 06 rigid/PSF | local pixels → predicted observation | `rigid_psf_forward.py`; NeSVoR RigidTransform/resolution2sigma |
| 07 training | dataset → Stage A/B checkpoint/log | `training_space.py`, `trad_trainer.py`; NeSVoR `inr/train.py` adaptation |
| 08 export | trained INR → physical RAS NIfTI/poses | `export_quantitative.py`; NeSVoR sample-volume coordinate principle |
| 09 QC | artifacts → finite/rigid-only report | `qc.py` |

训练空间严格复用 NeSVoR centering/scaling：physical RAS query 的唯一入口是
`(x_ras_mm-center_ras_mm)/30`。group pose 从 physical DICOM RAS 转为 training
space 时 compose `-center`、translation 除以 30；导出时反向还原。PSF 始终先在
scaled local slice axes 采样，再作 group rigid transform。

## v1 objective 与禁用项

data term 是 10-weight 平衡 MSE；训练前仅从完整 subject 的全部 SAX/2CH/4CH masked
intensities 计算一个 robust trimmed-mean scalar（q10=0.1、q90=0.9、严格取两者
之间的值的均值）。loss 比较 normalized prediction 与 raw observation/该 scalar，
因而所有 weight/stack/group 的相对强度保持不变；训练中的 A 为 normalized units。
导出的 `amplitude_3D` 乘回该 scalar，恢复原始输入 intensity units；T1/T2/B1 单位不变。
该 scalar 与训练单位写入 resolved config 和 checkpoint。Stage A 冻结 rigid，Stage B 在同一模型
上开放 rigid，并以相对 `axisangle_init` 的 NeSVoR `trans_loss` 正则。T1/T2/B1
field regularization 使用范围归一化后的 gradient，权重全部来自 YAML。

v1 明确没有 deform、EPG、per-weight/slice scale、bias、pixel/slice variance、
low-rank 或额外模型。`A(x)` 是唯一共同信号幅度。

## CPU tiny smoke

仅使用已有 Phase-2 `CYJ / 2ch / 1 group / 10 weights` MAT 重新 bridge；不运行
MIND 或 MP-PCA。输出 root 必须不存在，避免覆盖。

```bash
cd /home/universe/SVR/multimap_postprogramming/Code_10w/trad
bash scripts/run_smoke_cpu.sh
```

该命令进行 14 次 CPU iteration（A=10、B=4，PSF K=2）并写入
`/tmp/multimap_phase4_trad_smoke/outputs/`。

## server 示例（本地不要执行）

```bash
cd /home/universe/SVR/multimap_postprogramming/Code_10w/trad
PREPARED_SAX_NPZ=/absolute/sax/observations.npz \
PREPARED_2CH_NPZ=/absolute/2ch/observations.npz \
PREPARED_4CH_NPZ=/absolute/4ch/observations.npz \
OUTPUT_DIR=/absolute/empty_output \
bash scripts/run_server_example.sh
```

该示例明确启动 SAX+2CH+4CH joint reconstruction；示例配置为
`configs/server_train_example.yaml`，数值只是显式示例，不宣称最优。

## 输出

- `T1_3D.nii.gz`、`T2_3D.nii.gz`、`B1_3D.nii.gz`、`amplitude_3D.nii.gz`：physical
  RAS-mm grid 的 quantitative fields；NIfTI affine 同样是 RAS-mm。
- `model.pt`：model state、resolved config、HHZ protocol、validated TR/VPS、
  physical/training bbox、center/scaling、initial/current group axis-angle 与 seed。
- `final_rigid_poses.json`：undo center/scaling 后的 physical RAS-mm poses，写明
  `world=R@(local+T)` convention。
- `training_log.csv`、`config_resolved.yaml`、`qc_report.json`：训练与 QC 记录。

## provenance 与待确认

`third_party/UPSTREAM.md` 固定 NeSVoR source commit；该目录没有 `.git`。HHZ
TI=50/150 与 DYZ=10/100 的差异已明确记录。scanner 的 startup 和真实 k-space
centre timing 后续仍可核对，但 v1 当前严格沿用 HHZ
`function_T1T2_10HB_bssfp.m`/`sim_T1T2_10HB_bssfp.m` 行为。

量化 field regularization 保持原来的空间 L2 gradient 含义，但用
`sqrt(sum(g^2) + 1e-12)` 避免零梯度反传奇异性；optimizer 在 step 前检查所有当前
可训练梯度有限，step 后检查参数有限，发现异常即明确失败而不做 clip、clamp 或填零。
