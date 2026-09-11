# Codex Prompt — MultiMap 10-weight Direct Quantitative SVR Full Pipeline v1

> **任务目标**：在本地一次性搭建两个彼此独立、结构平行的 MultiMap 10-weight direct quantitative SVR 方法代码库：
>
> 1. `trad`：不训练信号模拟 MLP，直接把现有 HHZ MATLAB 中的 `sim_T1T2_10HB_bssfp.m` 严格等价改写为可微、批量化 PyTorch signal simulator，并嵌入 Quantitative NeSVoR。
> 2. `mlp`：使用与 `trad` 完全相同的物理 signal simulator 作为 teacher，离线训练一个 mDM 风格的跨受试者 MLP signal simulator，再将冻结后的 MLP 嵌入同一个 Quantitative NeSVoR。
>
> 两个方法除 signal decoder 外，其他数据处理、INR、rigid motion、PSF、loss、训练调度和输出必须尽量一致，以便后续做公平比较。
>
> **本任务不要做 EPG，不要做 deformable NeSVoR，不要引入新的深度网络方案，不要擅自改变序列物理模型。**

---

# 0. 必须先理解的项目背景和科学约束

本项目目标不是传统流程：

```text
10 weighted images
→ 2D dictionary matching
→ 2D T1/T2
→ NeSVoR
→ 3D T1/T2
```

而是直接建立：

```text
10-weighted 2D slices from SAX/2CH/4CH
        ↓
continuous 3D quantitative fields
T1(x), T2(x), B1(x), A(x)
        ↓
MultiMap signal forward model
        ↓
10 continuous weighted signal fields
        ↓
NeSVoR rigid motion + slice PSF
        ↓
predicted 2D weighted observations
        ↓
compare with real 2D weighted observations
        ↓
self-supervised joint optimization
        ↓
3D T1 / T2
```

核心 forward model：

\[
x
\rightarrow
q(x)=\{T_1(x),T_2(x),B_1(x),A(x)\}
\rightarrow
S_m(x)
\rightarrow
\hat I_{g,m,j}
\]

其中：

\[
S_m(x)
=
A(x)\,F_m(T_1(x),T_2(x),B_1(x),\tau_g)
\]

并且对每个 observed pixel 使用 NeSVoR 的 PSF Monte-Carlo sampling：

\[
\hat I_{g,m,j}
=
\frac{1}{K}
\sum_{k=1}^{K}
A(x_{g,m,j,k})
F_m(
T_1(x_{g,m,j,k}),
T_2(x_{g,m,j,k}),
B_1(x_{g,m,j,k}),
\tau_g
)
\]

注意顺序必须是：

```text
query quantitative parameters at each PSF sample
→ signal simulation at each PSF sample
→ average signals over PSF samples
```

**不要**先平均 T1/T2/B1 再做 signal simulation。

---

# 1. 不可违反的总要求

## 1.1 工作目录

所有新代码只允许创建在：

```text
/home/universe/SVR/multimap_postprogramming/Code_10w
```

创建两个独立子目录：

```text
/home/universe/SVR/multimap_postprogramming/Code_10w/trad
/home/universe/SVR/multimap_postprogramming/Code_10w/mlp
```

两个方法要做到**可单独复制、单独运行、单独同步**。

不要依赖 `Code_10w` 之外的新写共享 Python 包；允许读取外部已有数据和旧代码，但正式方法运行所需代码应复制/整理进各自目录。

## 1.2 不允许破坏旧工程

以下目录只读参考，不删除、不重命名、不覆盖：

```text
/home/universe/SVR/multimap_postprogramming/MultiMapCode
/home/universe/SVR/multimap_postprogramming/subject_dicom
```

GitHub 中用户自己的：

```text
universe26dyz/multimap-svr
```

优先参考：

```text
Multimapcode_hhz
```

其次只参考 `multimapcode_dyz` 中的自动化、MP-PCA 和“让 MIND 结果真正进入后续流程”的修正。

**HHZ 的科学模型是当前 v1 的权威来源。**

## 1.3 不做 EPG

本版本禁止使用 EPG 作为 teacher 或在线 physics：

```text
不要复制/使用 EPG_GRE_dict.m 作为正式 signal teacher
不要训练 EPG surrogate
不要把 mDM 的 EPG 参数替代 HHZ 参数
```

`mDM` 只用于参考其 **MLP 架构、训练方式、测试方式**。

## 1.4 第一版只做 rigid

第一版：

```text
MIND：INR 前处理 10-weight intra-slice motion
NeSVoR rigid：INR 中的 slice-to-volume / stack-to-volume motion
```

禁止：

```text
--deformable
DeformNet
deform embedding
deform loss
B-spline FFD
```

这些以后再做，不属于 v1。

## 1.5 尽量复用原始源码

NeSVoR 和 mDM 能直接沿用的代码必须优先基于原代码修改，不要重新凭空实现同类模块。

允许将上游源码复制到每个方法的：

```text
third_party/
```

但 `third_party` 只能包含普通文件：

```text
.py
.m
.md
LICENSE
...
```

**绝对不能包含 `.git/`，不能形成 nested git repo。**

若需要下载上游仓库：

1. 可先 clone 到 `/tmp/...`；
2. 再复制普通文件进入 `third_party/`；
3. 确保 `find <method>/third_party -name .git` 无结果；
4. 在 `third_party/UPSTREAM.md` 写清：
   - upstream URL
   - branch
   - commit SHA
   - 复制了哪些文件
   - 哪些项目代码在其基础上修改
5. 保留上游源码原有 copyright/header；
6. 若上游存在 LICENSE，则复制 LICENSE；若没有 LICENSE，不要自行虚构。

## 1.6 本地环境和计算限制

本地 Conda 环境：

```text
knesvr_torch
```

本地只有 CPU。

本地只允许：

- syntax/import check；
- unit test；
- MATLAB/Python signal parity 的小样本；
- 1 个受试者、1 个 stack、1–2 个 spatial slices、每个 slice 保留完整 10 weights 的 tiny smoke；
- MLP 的极小 synthetic training smoke；
- 几十 iterations 的 end-to-end smoke。

禁止在本地：

- 正式 NeSVoR 长训练；
- 大规模 MLP training；
- 全部 5 个受试者全流程；
- 大批量 MIND 重跑；
- 数千/数万 iteration reconstruction。

正式训练稍后由用户复制到服务器后运行。

Python 命令示例优先写为：

```bash
conda run -n knesvr_torch python ...
```

如本机存在 MATLAB，可以做极小 MATLAB parity；如不存在 MATLAB，不要因此阻塞整个代码搭建，保留测试脚本和运行说明，并使用已有 MAT/少量预处理结果完成能完成的 Python smoke。

## 1.7 本地数据

数据目录：

```text
/home/universe/SVR/multimap_postprogramming/subject_dicom
```

其中有 5 个受试者。

搭建和测试阶段只自动选择：

```text
1 个受试者
1 个 stack
1–2 个 spatial slices
每个 spatial slice 必须保留全部 10 weights
```

不要为了 smoke 遍历全部数据。

选择规则要确定性，例如：

```text
按目录名字典序选择第一个可用受试者和第一个可用 stack
```

在日志中写清实际选择了什么。

---

# 2. 当前 HHZ 序列/后处理模型：必须保持一致

优先读取用户 GitHub：

```text
https://github.com/universe26dyz/multimap-svr
```

重点：

```text
Multimapcode_hhz/PreData.m
Multimapcode_hhz/PostData.m
Multimapcode_hhz/Utility/function_T1T2_10HB_bssfp.m
Multimapcode_hhz/Utility/sim_T1T2_10HB_bssfp.m
Multimapcode_hhz/Utility/HHMMSS2Sec.m
Multimapcode_hhz/MIND_registration/MIND_descriptor2D.m
Multimapcode_hhz/MIND_registration/deformableReg2Dmind_asym_nodisplay.m
```

当前 v1 采用 HHZ 中的科学参数：

```text
NumImg = 10
FA = [45, 45, 45] degree
TI = [50, 150] ms
T2_prep = [35, 45, 55] ms
TR = DICOM Info.RepetitionTime
VPS / n_ex = DICOM Info.EchoTrainLength
nRampUp = 10
```

10-heartbeat topology：

```text
HB1   inversion, TI1
HB2   recovery
HB3   recovery
HB4   recovery

HB5   inversion, TI2
HB6   recovery
HB7   recovery

HB8   T2prep1
HB9   T2prep2
HB10  T2prep3
```

HHZ 的 B1 定义：

```text
effective flip angle = nominal FA × B1
```

传统 parameter ranges：

```text
T1: 20–2500 ms
T2: 5–200 ms
B1: 0.1–1.2
constraint: T1 > T2
```

## 2.1 必须处理 HHZ 与 DYZ 的 TI 差异

当前 DYZ `PreData.m` 中曾出现：

```text
TI = [10, 100]
```

而 HHZ 是：

```text
TI = [50, 150]
```

本任务 v1 以 HHZ 为权威：

```text
TI = [50, 150]
```

不要无声继承 DYZ 的 `[10,100]`。

在 README 的“已知差异/待核对项”中明确记录这个差异。

## 2.2 MIND 必须真的进入后续数据

HHZ 原始 `PreData.m` 计算了：

```text
MIND_mag_reg
```

但原脚本保存的是未注册的 `Mag_crop`。

DYZ 做过正确修正：

```matlab
Mag_crop = MIND_mag_reg;
```

v1 要沿用这个修正。

但除这个数据流修正、自动化、MP-PCA 外，不要把 DYZ 的其他科学参数自动覆盖 HHZ。

---

# 3. 两个方法的最终目录结构

两个目录结构尽量一致，只让 Module 05 signal decoder 有差异。

## 3.1 `trad`

```text
Code_10w/trad/
├── readme.md
├── changelog.md
├── requirements_note.md
├── configs/
│   ├── protocol_hhz_v1.yaml
│   ├── smoke_cpu.yaml
│   └── server_train_example.yaml
├── third_party/
│   ├── UPSTREAM.md
│   └── nesvor/
│       ├── LICENSE
│       ├── README.md
│       └── nesvor/...
├── modules/
│   ├── module_01_preprocess_matlab/
│   │   ├── hhz_original/
│   │   ├── preprocessing_v1/
│   │   └── README_module.md
│   ├── module_02_data_bridge/
│   ├── module_03_dataset_geometry/
│   ├── module_04_quantitative_inr/
│   ├── module_05_signal_decoder/
│   ├── module_06_rigid_psf/
│   ├── module_07_objective_training/
│   ├── module_08_inference_export/
│   └── module_09_qc_benchmark/
├── scripts/
│   ├── prepare_smoke_case.sh
│   ├── run_smoke_cpu.sh
│   └── run_server_example.sh
├── tests/
└── outputs/
    └── .gitkeep
```

## 3.2 `mlp`

```text
Code_10w/mlp/
├── readme.md
├── changelog.md
├── requirements_note.md
├── configs/
│   ├── protocol_hhz_v1.yaml
│   ├── mlp_smoke_cpu.yaml
│   ├── mlp_server_train_example.yaml
│   ├── smoke_cpu.yaml
│   └── server_train_example.yaml
├── third_party/
│   ├── UPSTREAM.md
│   ├── nesvor/
│   │   ├── LICENSE
│   │   ├── README.md
│   │   └── nesvor/...
│   └── mdm/
│       ├── README_upstream.md
│       └── fast_dictionary_generation_reference/
│           ├── Models/signal_simulation.py
│           ├── train_ss_net.py
│           └── test_ss_net.py
├── modules/
│   ├── module_01_preprocess_matlab/
│   ├── module_02_data_bridge/
│   ├── module_03_dataset_geometry/
│   ├── module_04_quantitative_inr/
│   ├── module_05_signal_decoder/
│   │   ├── trad_teacher/
│   │   ├── mlp_model.py
│   │   ├── synthetic_dataset.py
│   │   ├── train_mlp.py
│   │   ├── test_mlp.py
│   │   └── frozen_decoder.py
│   ├── module_06_rigid_psf/
│   ├── module_07_objective_training/
│   ├── module_08_inference_export/
│   └── module_09_qc_benchmark/
├── scripts/
│   ├── train_mlp_smoke_cpu.sh
│   ├── prepare_smoke_case.sh
│   ├── run_smoke_cpu.sh
│   └── run_server_example.sh
├── tests/
└── outputs/
    └── .gitkeep
```

如某些 Python 包必须增加 `__init__.py`，直接增加。

不要把大 DICOM、训练集、checkpoint、NIfTI 结果复制进代码仓库；outputs 只留轻量日志/示例或 `.gitkeep`。

---

# 4. 上游 NeSVoR 和 mDM 源码复用要求

## 4.1 NeSVoR

上游：

```text
https://github.com/daviddmc/NeSVoR
branch: master
```

重点复用：

```text
nesvor/inr/
    data.py
    hash_grid_torch.py
    models.py
    sample.py
    train.py

nesvor/image/
    __init__.py
    image.py
    image_utils.py

nesvor/transform/
    __init__.py
    transform.py
    transform_convert.py
    transform_convert_torch.py

nesvor/utils/
    __init__.py
    logger.py
    loss.py
    misc.py
    psf.py
    types.py
```

以及为了 import closure 所需的少量其他纯 Python 文件。

允许直接把整个 `nesvor/` Python package 复制进 `third_party/nesvor/nesvor/`，如果这样比人工追依赖更可靠；但不要复制 `.git`。

项目正式的 Quantitative INR/训练代码不要重新从零写 NeSVoR 同类实现；应：

1. import 上游类；
2. 或复制 `models.py/train.py/data.py` 的相关类为 project-local adapted class；
3. 保留“源自 NeSVoR”的注释；
4. 在代码注释和 `UPSTREAM.md` 写清具体修改点。

必须尽量保留 NeSVoR 原始：

```text
HashGrid / HashEmbedder
RigidTransform
axis-angle rigid optimization
spatial centering/scaling
PSF sigma calculation
Monte-Carlo PSF sampling
AdamW optimizer style
learning-rate schedule
sample/export logic
```

## 4.2 mDM

上游：

```text
https://github.com/SJTU-CMRLab/Model-Based-MoCo-for-Cardiac-Multi-Parametric-Mapping
branch: master
```

只需要 mDM fast dictionary generation 中和 MLP 相关的模板：

```text
DM_LR_with_fast_dictionary_generation/Models/signal_simulation.py
DM_LR_with_fast_dictionary_generation/train_ss_net.py
DM_LR_with_fast_dictionary_generation/test_ss_net.py
```

可以再复制相关 README 片段作为 provenance。

**不要复制 EPG 作为正式模型。**

我们的 MLP 只参考其网络和训练结构，不使用其：

```text
T1/T2/9RR → EPG signal
```

teacher。

我们的 teacher 必须是 HHZ signal model 的 PyTorch 等价实现。

---

# 5. Module 01 — MATLAB preprocessing（两个方法相同）

目录：

```text
modules/module_01_preprocess_matlab/
```

## 输入

```text
/home/universe/SVR/multimap_postprogramming/subject_dicom/<subject>/<stack>/*.dcm
```

## 输出

每个 stack 的预处理 MAT，至少包含：

```text
Mag                  原始按时间排序后 [Nx,Ny,10,Nslice]
Mag_crop             实际供后续使用的 MIND-registered data
Acq_time             [10,Nslice], ms
TR
VPS
FA
TI
T2_prep
Info / 可恢复几何的必要元数据
```

## 功能

严格基于 HHZ：

```text
DICOM read
→ AcquisitionTime sorting
→ reshape [Nx,Ny,10,Nslice]
→ optional MP-PCA
→ MIND: weight 2–10 → weight 1
→ Mag_crop = MIND_mag_reg
→ save MAT
```

## 源码来源

复制：

```text
Multimapcode_hhz/PreData.m
Multimapcode_hhz/MIND_registration/*
Multimapcode_hhz/Utility/HHMMSS2Sec.m
Multimapcode_hhz/Utility/imshow3.m
```

MP-PCA 可复制 DYZ：

```text
multimapcode_dyz/matlab/preprocessing/Run_MPPCA_All_Stacks.m
multimapcode_dyz/matlab/preprocessing/mppca_denoise_multimap_stack.m
multimapcode_dyz/matlab/preprocessing/Check_MPPCA_Result.m
```

## 修改原则

创建：

```text
hhz_original/
```

保存 HHZ 原文件不修改。

创建：

```text
preprocessing_v1/
```

在副本上做：
- 路径参数化；
- 可关闭 figure；
- 可选择 1–2 slices 做 smoke；
- 让 `Mag_crop = MIND_mag_reg`；
- 保存必要 metadata；
- HHZ 的 TI/FA/T2prep 不改。

不要将 MIND 改写成 Python。

## MIND 后几何

因为 HB2–HB10 已经被 warp 到 HB1 的二维 grid：

```text
同一 spatial slice 的 10 weights 后续必须使用 HB1 的 slice geometry
```

此规则必须写进 README 和 data bridge。

---

# 6. Module 02 — MATLAB/Python data bridge（两个方法相同）

目录：

```text
modules/module_02_data_bridge/
```

目标：把 MATLAB 预处理结果和原始 DICOM geometry 转换成 Python quantitative SVR 可直接读取的统一数据格式。

至少实现：

```text
build_manifest.py
inspect_preprocessed_mat.py
prepare_observations.py
```

## 输入

- preprocessing MAT；
- 原始 DICOM 目录；
- `protocol_hhz_v1.yaml`。

## 输出

建议：

```text
prepared_case/
├── observations.h5 或 observations.npz
├── manifest.json
├── timing.npy
└── qc_summary.json
```

不要把大数据复制到源码目录，prepared_case 默认输出到用户指定工作目录。

## 每个 observation 必须包含

```text
group_id              一个 native spatial slice
weight_id             0–9 / HB1–HB10
stack_id              sax / 2ch / 4ch
image                  MIND-registered 2D image
mask                   foreground mask，可先采用 HHZ 的简单 mask 逻辑
affine / transform     HB1 geometry
resolution_xyz
slice_thickness
acquisition_time
timing_vector_id
```

## group 定义

一个 spatial slice：

```text
group g
├── weight 0
├── weight 1
...
└── weight 9
```

MIND 后，同一 group 的 10 weights 共用几何。

## timing

严格复用 `function_T1T2_10HB_bssfp.m` 的逻辑，建立一个明确 Python function：

```python
compute_duration_before_acq(
    acquisition_times_ms,
    tr_ms,
    n_ex,
    ti_ms,
    t2prep_ms,
) -> np.ndarray  # shape [10]
```

在线 MLP/Trad conditioning 使用：

```text
Duration_befor_Acq[1:10]
```

即 9 个有效 inter-HB conditioning values。

写 unit test，用固定手算例子验证。

---

# 7. Module 03 — Quantitative dataset + geometry（两个方法相同）

目录：

```text
modules/module_03_dataset_geometry/
```

不要重新自由设计一套完全不同于 NeSVoR 的 dataset。

以：

```text
third_party/nesvor/nesvor/inr/data.py::PointDataset
```

为基础修改。

建立 project-local：

```text
QuantPointDataset
```

## 每个 sampled pixel/batch 至少返回

```python
{
    "xyz": ...,          # observed slice pixel 3D coordinate before rigid transform
    "v": ...,            # observed intensity
    "group_idx": ...,    # native spatial slice group
    "weight_idx": ...,   # 0..9
    "stack_idx": ...,
    "timing": ...,       # 9D or index to group timing tensor
}
```

## 关键区别

原 NeSVoR：

```text
one slice_idx → one rigid pose
```

我们的 v1：

```text
one group_idx → one rigid pose
10 weights in the group share this pose
```

因为 MIND 已经完成 intra-group registration。

Rigid transform 初始化、bounding box、spatial centering/scaling 尽量复用 NeSVoR 的原实现。

## sampling balance

训练 batch 不要被某个 weight 或 SAX 的 pixel 数完全支配。

实现明确的 balanced sampling 或 balanced loss aggregation：

```text
10 weights 尽量均衡
SAX / 2CH / 4CH 尽量均衡
```

不要为此改变观测强度。

---

# 8. Module 04 — Quantitative INR（两个方法相同）

目录：

```text
modules/module_04_quantitative_inr/
```

必须以：

```text
NeSVoR nesvor/inr/models.py::INR
```

的 HashGrid 和网络构建方式为基础。

原 NeSVoR：

```text
x → HashGrid → density_net → scalar density V(x)
```

修改为：

```text
x → same HashGrid style → shared latent → T1/T2/B1/A
```

## 输出

```python
{
    "t1_ms": ...,
    "t2_ms": ...,
    "b1": ...,
    "amplitude": ...,
}
```

## 参数范围

为了贴合 HHZ dictionary 物理约束，v1 使用连续参数化：

```text
T2 in (5, 200) ms
T1 in (T2, 2500) ms
B1 in (0.1, 1.2)
A > 0
```

建议明确实现：

```python
t2 = 5.0 + 195.0 * sigmoid(raw_t2)
t1 = t2 + (2500.0 - t2) * sigmoid(raw_t1_gap)
b1 = 0.1 + 1.1 * sigmoid(raw_b1)
A = softplus(raw_A) + eps
```

这是项目对原 NeSVoR scalar output 的必要定量改造，不要改成其他范围。

## source reuse

优先复用：

```text
build_encoding
build_network
HashEmbedder
compute_resolution_nlevel
```

不要另写 positional encoding。

## v1 不做

```text
DeformNet
slice-specific unrestricted scale
slice-specific bias
pixel variance
slice variance
```

可保留上游源码在 `third_party`，但 project config 中默认全部关闭。

---

# 9. Module 05A — Trad direct signal decoder

只在：

```text
trad/modules/module_05_signal_decoder/
```

实现正式 Trad decoder。

## 唯一科学来源

```text
Multimapcode_hhz/Utility/sim_T1T2_10HB_bssfp.m
```

不要根据论文重新推导一个“看起来类似”的 Bloch model。

要求做到：

```text
MATLAB behavior parity first
vectorization second
optimization third
```

## 接口

至少定义：

```python
class TradSignalSimulator(torch.nn.Module):
    def forward(
        self,
        t1_ms: torch.Tensor,
        t2_ms: torch.Tensor,
        b1: torch.Tensor,
        timing9_ms: torch.Tensor,
        protocol: ProtocolConfig,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        return shape [..., 10]
        """
```

## 物理步骤必须逐项对应 HHZ

保留：

```text
M0 = 1
Mz initial = 1
10 heartbeats
FA block 1/2/3 × B1
perfect inversion Mz = -Mz
T1 recovery formula
T2prep as Mz *= exp(-TE/T2)
10 ramp-up RF pulses
alternating Dirc
2×2 [Mxy,Mz] rotation
E1 = exp(-TR/T1)
E2 = exp(-TR/T2)
n_ex readout pulses
central readout window averaging
Mz carried to next heartbeat
```

不要变成 EPG。

## vectorization

可以保留：

```text
10 HB loop
nRampUp loop
n_ex pulse loop
```

但 batch dimension 必须 tensorized。

禁止：

```text
for voxel in voxels:
    simulate_one_voxel(...)
```

## 输出

同时提供：

```text
raw_signal
normalized_signal = raw_signal / ||raw_signal||2
```

主 Quantitative SVR v1 使用：

```text
normalized_signal × A(x)
```

这样 signal decoder 只负责 fingerprint shape，A(x) 负责共同幅值。

## parity test

写：

```text
tests/test_trad_signal_parity.py
```

如果 MATLAB 可用：

- 随机 10–50 组 T1/T2/B1/timing；
- MATLAB HHZ simulator 作为 reference；
- Python output 对比。

若 MATLAB 不可用：
- 创建一个可由用户在有 MATLAB 环境运行的 `export_matlab_signal_reference.m`；
- 保存 deterministic test vectors；
- Python test 读取 reference；
- 本地至少验证 shape、finite、gradient、edge cases。

不要用旧 hard dictionary matching 的最终 T1/T2 map 作为唯一 parity 标准，优先比 signal 本身。

---

# 10. Module 05B — MLP signal decoder

只在：

```text
mlp/modules/module_05_signal_decoder/
```

实现。

## 10.1 teacher

先复制 `trad` 的 `TradSignalSimulator` 到：

```text
mlp/modules/module_05_signal_decoder/trad_teacher/
```

保持和 Trad 一致。

**MLP teacher 就是这个 HHZ-compatible Trad simulator。**

禁止 EPG teacher。

## 10.2 输入

固定 protocol 下，MLP 输入 12 维：

```text
T1
T2
B1
timing_2
timing_3
...
timing_10
```

normalized input：

```text
T1 / 1000
T2 / 1000
B1 原值
timing / 1000
```

不要把 B1 `/1000`。

## 10.3 输出

```text
10 normalized signal values
```

shape：

```text
[..., 10]
```

## 10.4 架构

严格参考 mDM：

```text
12
→ Linear(200)
→ BatchNorm
→ LeakyReLU
→ Linear(200)
→ BatchNorm
→ LeakyReLU
→ Linear(200)
→ BatchNorm
→ LeakyReLU
→ Linear(10)
```

在此基础上只做必要的 11→12 input 修改。

不要引入：
- Transformer；
- SIREN；
- residual MLP；
- attention；
- convolution。

网络代码尽量从上游：

```text
DM_LR_with_fast_dictionary_generation/Models/signal_simulation.py
```

复制后最小修改。

## 10.5 synthetic training data

创建：

```text
synthetic_dataset.py
```

连续随机采：

```text
T1 ∈ [20,2500]
T2 ∈ [5,200]
B1 ∈ [0.1,1.2]
且 T1 > T2
```

timing：
- smoke 时只使用选定 smoke subject 中少量真实 9D timing vectors；
- 正式代码支持读取任意 `timing_pool.npy`；
- 不要在本地为了建 timing pool 扫完全部 5 受试者；
- server formal training 可以由用户以后提供更大的 timing pool。

teacher：

```text
TradSignalSimulator
```

生成：

```text
10D normalized target signal
```

## 10.6 数据规模配置

不要把数据量写死。

`mlp_smoke_cpu.yaml`：

```text
train: 4096
val: 1024
test: 1024
epochs: 2–5
batch_size: 128 或更小
```

只验证程序可运行。

`mlp_server_train_example.yaml`：

```text
可配置到百万级；
给出 12.5M 作为“参考 mDM 的规模示例”，但不要本地自动运行。
```

## 10.7 train

训练方式尽量参考 mDM：

```text
MSE
Adam
initial LR ~1e-3
validation
best checkpoint
```

保存：

```text
signal_simulator_best.pth
normalization.json
train_history.csv
test_metrics.json
```

## 10.8 frozen decoder

在线 Quantitative SVR：

```python
for p in simulator.parameters():
    p.requires_grad = False
```

但是**禁止**用 `torch.no_grad()` 包住 signal forward。

必须允许：

```text
loss
→ signal
→ frozen MLP
→ T1/T2/B1
→ Quantitative INR
```

的 input gradient。

写 test 验证：

```python
t1.requires_grad = True
signal = frozen_decoder(...)
signal.sum().backward()
assert t1.grad is finite
```

---

# 11. Module 06 — NeSVoR rigid motion + PSF（两个方法相同）

目录：

```text
modules/module_06_rigid_psf/
```

以 NeSVoR：

```text
nesvor/inr/models.py::NeSVoR
nesvor/transform::RigidTransform
resolution2sigma
PSF random sampling
ax_transform_points
```

为主干。

## 11.1 rigid only

模型只维护：

```text
axisangle[group_idx]
```

一个 spatial slice group 一个 6-DOF rigid transform。

同一 group 10 weights 共用 rigid pose。

不要创建 10 个独立 rigid pose，因为 MIND 已经把 intra-group 10 weights 配到 HB1。

## 11.2 pose initialization

尽量复用 NeSVoR 原有：
- DICOM/NIfTI geometry；
- stack registration / initial transforms；
- centering/scaling；
- transformation regularization。

如果直接调用 NeSVoR CLI 的 stack registration 很难和 10-weight group manifest 对接，允许写 project adapter，但 rigid math 仍使用 `RigidTransform`，不要自建 Euler-angle system。

## 11.3 PSF

复用 NeSVoR：

```text
resolution2sigma
anisotropic Gaussian PSF
n_samples Monte-Carlo points
```

输入：

```text
xyz observed pixel coordinate
group rigid transform
slice resolution / thickness
```

输出：

```text
xyz_psf: [batch, K, 3]
```

然后送入 Quantitative INR + signal decoder。

## 11.4 禁止 deform

配置和源码中确保：

```text
deformable = false
```

不建立：
- `deform_embedding`
- `DeformNet`
- `D_REG`

---

# 12. Module 07 — Objective and training（两个方法尽量相同）

目录：

```text
modules/module_07_objective_training/
```

以 NeSVoR `inr/train.py` 为训练骨架：

```text
PointDataset style batch
model forward
AdamW
scheduler
logging
checkpoint
```

## 12.1 model forward

一次 batch：

```text
observed xyz
group_idx
weight_idx
timing
observed v
```

执行：

```text
1. NeSVoR-style rigid + PSF sample coordinates
2. query Quantitative INR at every PSF coordinate
3. run Trad or MLP signal decoder
4. select current weight m
5. multiply by A(x)
6. PSF average
7. compare with observed v
```

## 12.2 v1 intensity freedom

为了防止 quantitative fingerprint 被 nuisance terms 吞掉：

默认：

```text
no per-weight scale
no per-slice scale
no bias network
no pixel variance
no slice variance
```

即不要直接沿用 NeSVoR 中：

```text
logit_coef
b_net
sigma_net
log_var_slice
```

到 v1 主模型。

这些上游代码可保留 third_party，但关闭。

## 12.3 data loss

v1 先用清楚可解释的 MSE：

\[
L_{data} = mean((I-\hat I)^2)
\]

实现 weight-balanced aggregation：

```text
先算每个 weight 的 mean loss
再 10 weights 平均
```

如 batch 同时来自多 stack，再做 stack balance，避免 SAX 单纯因为 slice 多而主导。

不要第一版加入 low-rank loss。

## 12.4 regularization

尽量复用 NeSVoR 的 image regularization 思路。

v1 至少支持：
- `T1` edge/TV regularization；
- `T2` edge/TV regularization；
- `B1` 更强的 smooth L2/TV；
- rigid transformation regularization，复用 NeSVoR `trans_loss` 的逻辑。

所有权重放 config，不写死。

默认 smoke 权重可以很小，确保 loss finite。

## 12.5 staged optimization

实现两个主要 stage：

### Stage A — quantitative warmup

```text
rigid pose frozen
optimize T1/T2/B1/A only
```

### Stage B — joint rigid

```text
unfreeze group rigid pose
optimize quantitative INR + rigid pose jointly
```

可选 Stage C 只作为配置：

```text
open finer hash resolution / longer training
```

不要实现 deform stage。

## 12.6 optimizer

尽量沿用 NeSVoR：

```text
AdamW
network params + encoding params分组
weight_decay for network
milestone LR scheduler
```

CPU smoke：
- single precision；
- tiny batch；
- tiny n_samples；
- 10–50 iterations。

server config 再提供正式大参数示例，但不要本地运行。

---

# 13. Module 08 — inference / export（两个方法相同）

目录：

```text
modules/module_08_inference_export/
```

复用 NeSVoR `sample.py` 和 image/Volume 相关代码。

## 输出必须至少包括

```text
T1_3D.nii.gz
T2_3D.nii.gz
B1_3D.nii.gz
amplitude_3D.nii.gz
model.pt
final_rigid_poses.json
training_log.csv
config_resolved.yaml
```

可选 QC：

```text
simulated_weighted_slices/
residual_weighted_slices/
```

输出分辨率默认支持 1.0 mm isotropic，但 smoke 可以更粗减少 CPU 时间。

---

# 14. Module 09 — QC / benchmark（非常重要）

两个方法都必须有。

## 14.1 preprocessing QC

检查：

```text
10 weights / group
AcquisitionTime strictly sorted
Mag_crop 确实为 MIND 结果
all 10 weights share HB1 geometry after MIND
TI = [50,150]
T2prep = [35,45,55]
FA = 45
```

## 14.2 Trad signal parity

至少测试：

```text
output shape
finite
T1/T2/B1 gradient finite
known deterministic cases
MATLAB parity when reference available
```

生成：

```text
trad_signal_parity.csv
```

## 14.3 MLP fidelity

MLP 方法测试：

```text
MLP signal vs Trad teacher
overall RMSE
per-weight RMSE for HB1...HB10
max error
```

并测试 gradient-through-frozen-MLP。

## 14.4 Trad vs MLP simulator benchmark

提供统一脚本：

```text
benchmark_signal_decoder.py
```

对 batch size：

```text
1e3
1e4
1e5
```

如 CPU 过慢则自动缩小，不强行跑完。

测：

```text
forward wall time
backward wall time
peak CPU memory（若方便）
GPU memory（服务器以后自动支持）
```

不要为了本地 benchmark 跑超长任务。

## 14.5 end-to-end tiny smoke

只用：

```text
1 subject
1 stack
1–2 spatial slices
all 10 weights
```

要求验证：

```text
dataset loads
forward works
backward works
loss finite
loss 至少在极短训练中不出现 NaN/Inf
rigid pose tensor exists
deform parameter does not exist
T1/T2/B1/A 3D sampler works
checkpoint save/load works
```

不要以“CPU smoke loss下降很多”作为成功条件；只要链路和梯度正确。

---

# 15. README 要求（每个方法单独）

每个：

```text
trad/readme.md
mlp/readme.md
```

必须用中文，按照数据流详细写。

至少包含：

1. 方法目的；
2. 与传统 `2D DM → NeSVoR` 的区别；
3. 完整 pipeline ASCII 图；
4. 每个 module：
   - 输入；
   - 输出；
   - 功能；
   - 关键代码文件；
   - 上游来源；
   - command 示例；
5. HHZ protocol 参数；
6. MIND 的位置和作用；
7. NeSVoR rigid + PSF 的复用点；
8. Trad/MLP signal decoder 的定义；
9. 为什么 v1 禁用：
   - deform；
   - per-weight scale；
   - bias；
   - variance；
   - low-rank；
10. CPU smoke 命令；
11. server 正式运行示例；
12. 输出文件说明；
13. 已知问题/待确认：
   - HHZ TI 50/150 vs DYZ 10/100；
   - scanner 真正 startup/k-space-center timing 后续仍可核对；
14. third_party provenance。

MLP README 额外写清：

```text
MLP 是跨受试者 universal simulator
不是 subject-specific model
不是 inverse T1/T2 estimator
不是 hard dictionary matching
online reconstruction 中不生成 dictionary
```

---

# 16. changelog.md 要求

每个方法根目录必须：

```text
changelog.md
```

只能 append，绝对不能覆盖历史内容。

第一次建立格式：

```markdown
# Changelog

## 2026-09-10 — Full pipeline v1 scaffold
- 新建了哪些模块
- 复制了哪些 upstream source
- 修改了哪些原始逻辑
- 科学参数是什么
- 运行了哪些测试
- 哪些测试未运行及原因
```

后续 Codex 每次代码修改都必须：
- 新增一段日期/时间；
- 列出改动文件；
- 说明原因；
- 写实际执行的测试和结果；
- 不改旧记录。

---

# 17. command 设计要求

两个项目都提供统一风格 CLI，至少支持：

## prepare / inspect

```bash
conda run -n knesvr_torch python -m modules.module_02_data_bridge.prepare_observations \
  --preprocessed-mat /path/to/data.mat \
  --dicom-dir /path/to/dicom \
  --stack sax \
  --output-dir /path/to/prepared_case \
  --max-groups 2
```

## Trad smoke

```bash
cd /home/universe/SVR/multimap_postprogramming/Code_10w/trad
bash scripts/run_smoke_cpu.sh
```

## MLP smoke training

```bash
cd /home/universe/SVR/multimap_postprogramming/Code_10w/mlp
bash scripts/train_mlp_smoke_cpu.sh
```

## MLP reconstruction smoke

```bash
bash scripts/run_smoke_cpu.sh
```

正式 server command 写到 README 和：

```text
scripts/run_server_example.sh
```

但是不要本地执行。

---

# 18. 测试要求

优先使用 `pytest`。

至少建立：

```text
tests/test_protocol.py
tests/test_timing.py
tests/test_dataset_grouping.py
tests/test_quantitative_inr.py
tests/test_rigid_group_pose.py
tests/test_psf_forward.py
tests/test_signal_decoder.py
tests/test_end_to_end_tiny.py
```

MLP 额外：

```text
tests/test_mlp_model_shape.py
tests/test_mlp_teacher_fidelity.py
tests/test_frozen_mlp_gradient.py
```

每个模块完成后立即跑对应小测试，不要等全部写完才测试。

---

# 19. 代码复用和“不要自由发挥”的具体约束

遇到已有 NeSVoR 功能时，优先顺序：

```text
1. 直接 import vendored NeSVoR source
2. subclass / wrapper
3. 复制上游函数后做最小修改
4. 最后才允许自己写新实现
```

以下功能不要自己重写：

```text
HashGrid encoding
RigidTransform / axis-angle transform
NeSVoR centering/scaling思路
resolution2sigma
PSF Gaussian sampling
NeSVoR sample-volume基础逻辑
mDM 3×200 MLP基本结构
```

必须自己新增/修改的只有项目必需差异：

```text
10-weight group metadata
Quantitative T1/T2/B1/A heads
HHZ signal simulator PyTorch parity version
MLP 12D conditioning
weight-specific signal selection
signal-before-PSF forward model
group-shared rigid pose
balanced 10-weight loss
quantitative outputs
```

不要新增与 v1 无关的复杂模块。

---

# 20. Trad 与 MLP 的公平性要求

两者必须尽量共享相同默认配置：

```text
same preprocessing
same MIND
same prepared data
same rigid initialization
same quantitative INR
same HashGrid
same PSF K
same batch size
same loss
same regularization
same optimizer
same LR schedule
same training stages
same output resolution
```

唯一核心变量：

```text
Trad:
HHZ-compatible differentiable direct signal simulator

MLP:
frozen MLP surrogate trained against the same Trad simulator
```

因此不要让：
- Trad 用 raw signal，MLP 用另一种 normalization；
- Trad 和 MLP 用不同 T1/T2/B1 ranges；
- 两边用不同 pose 数；
- 两边用不同 loss；
- 两边一个开 variance、另一个不开。

---

# 21. 正式代码完成后的验收标准

本地只要求“小而完整”的证据。

最终必须确认：

```text
[ ] Code_10w/trad 存在且独立
[ ] Code_10w/mlp 存在且独立
[ ] 两者均有 readme.md
[ ] 两者均有 append-only changelog.md
[ ] third_party 无任何 .git
[ ] UPSTREAM.md 记录 commit SHA 和来源
[ ] HHZ MATLAB 原代码副本保留
[ ] v1 preprocessing 使用 MIND 输出
[ ] v1 protocol 使用 TI 50/150
[ ] Trad PyTorch simulator 可微
[ ] Trad simulator 有 parity/QC
[ ] MLP 结构为 12→200→200→200→10
[ ] MLP teacher 为 Trad，不是 EPG
[ ] frozen MLP input gradient 可通过
[ ] Quantitative INR 输出 T1/T2/B1/A
[ ] T1>T2 由参数化保证
[ ] rigid transform 一个 group 一个，10 weights 共享
[ ] deform 完全关闭
[ ] PSF 来自 NeSVoR 思路/源码
[ ] no per-weight scale
[ ] no bias/variance default v1
[ ] 10-weight loss balanced
[ ] tiny CPU end-to-end smoke 可执行
[ ] 无 NaN/Inf
[ ] server scripts 已生成但未本地正式运行
```

---

# 22. 执行顺序

严格按以下顺序执行，不要一上来同时大改两个方法：

### Step 1 — source/protocol audit
- 检查本地/GitHub HHZ 和 DYZ；
- 确认上述文件；
- 建 `protocol_hhz_v1.yaml`；
- 记录 TI 差异；
- 不修改旧工程。

### Step 2 — scaffold `trad`
- 建目录；
- vendor NeSVoR；
- copy MATLAB；
- 写 data bridge；
- 写 dataset；
- 改 Quantitative INR；
- 写 TradSignalSimulator；
- 做 signal parity；
- 接 rigid + PSF；
- 接训练；
- tiny smoke。

### Step 3 — freeze Trad scientific contract
只有 Trad signal parity 和 tiny forward/backward 通过后，再开始 MLP。

### Step 4 — scaffold `mlp`
- 从 Trad 复制共同模块；
- vendor 同一 NeSVoR snapshot；
- vendor mDM MLP reference；
- 保证共同模块的源码版本一致。

### Step 5 — MLP offline branch
- copy Trad teacher；
- 建 synthetic dataset；
- 改 mDM MLP 11→12 input；
- tiny CPU MLP training；
- fidelity + gradient tests。

### Step 6 — MLP online integration
- 用 FrozenMLPSignalDecoder 替换 Trad decoder；
- 其他模型保持不变；
- tiny end-to-end smoke。

### Step 7 — benchmark
- Trad vs MLP signal forward/backward；
- 输出轻量 CSV/JSON；
- 不在 CPU 跑大 benchmark。

### Step 8 — docs
- 完成两个 readme.md；
- append changelog.md；
- 完成 `Code_10w/IMPLEMENTATION_REPORT_v1.md`。

---

# 23. 最终报告

任务完成后生成：

```text
/home/universe/SVR/multimap_postprogramming/Code_10w/IMPLEMENTATION_REPORT_v1.md
```

中文，至少说明：

1. 最终目录树；
2. Trad 完成了什么；
3. MLP 完成了什么；
4. 具体 vendor 了哪些 NeSVoR/mDM 文件和 commit；
5. HHZ vs DYZ 检查结果；
6. 实际 smoke 使用了哪个 subject/stack/slice；
7. 实际执行过的命令；
8. 每项 test 的 PASS/FAIL；
9. MLP tiny train 的结果；
10. Trad vs MLP tiny benchmark；
11. 哪些测试因为本地无 GPU/无 MATLAB 未运行；
12. 到服务器后推荐的第一组正式命令；
13. 当前仍需用户核对的科学问题，但不要因为这些问题阻塞 v1 代码搭建。

---

# 24. 额外工程纪律

- 不删除任何旧文件。
- 不在 `third_party` 中保留 `.git`。
- 不初始化 `trad/.git` 或 `mlp/.git` nested repositories。
- 不自动下载/生成巨大数据。
- 不跑全 5 受试者。
- 不跑正式 GPU training。
- 不引入 EPG。
- 不打开 deformable。
- 不加入 low-rank loss。
- 不改变 HHZ protocol 参数来“让结果更好”。
- 遇到源码与本 prompt 冲突，以本 prompt 的 v1 科学约束为准，并在 report 中记录冲突。
- 遇到工程小问题自行解决，不要因为非科学性小问题频繁中断询问。
- 如果某个科学参数无法从现有代码确认，保持 HHZ 当前行为，不自行猜值，并记录到 `IMPLEMENTATION_REPORT_v1.md`。
- 所有运行都要有日志；长命令不要高频轮询。
- 正式功能完成前不要做无关重构或代码美化。

---

# 25. 最终要达到的两个方法

## Trad

\[
\boxed{
\text{MIND 10w preprocessing}
\rightarrow
\text{Quantitative NeSVoR}
\rightarrow
\text{HHZ direct differentiable simulator}
\rightarrow
\text{NeSVoR rigid + PSF}
\rightarrow
\text{2D data consistency}
\rightarrow
3D\ T1/T2
}
\]

## MLP

离线：

\[
\boxed{
\text{HHZ Trad simulator}
\rightarrow
\text{synthetic signals}
\rightarrow
\text{mDM-style universal MLP}
}
\]

在线：

\[
\boxed{
\text{MIND 10w preprocessing}
\rightarrow
\text{Quantitative NeSVoR}
\rightarrow
\text{frozen MLP simulator}
\rightarrow
\text{NeSVoR rigid + PSF}
\rightarrow
\text{2D data consistency}
\rightarrow
3D\ T1/T2
}
\]

两者最终应能够用同一 prepared case 和近似同一 command 配置进行对比。

请现在按以上顺序自动完成代码搭建、最小测试、文档和最终报告。不要进行正式大规模训练。
