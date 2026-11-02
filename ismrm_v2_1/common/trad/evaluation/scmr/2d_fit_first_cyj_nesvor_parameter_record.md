# CYJ 2D-fit-first（SCMR）NeSVoR 两组参数记录

## 结论：训练输入是否做了 crop

**没有做空间 crop。** NeSVoR 实际使用的是 `fitting_qc/NIfTI_Slices` 中的六个 mapping stack。MATLAB 预处理变量虽命名为 `Mag_crop`，但 `PreData.m` 中明确执行的是 `Mag_crop = Mag`；随后只进行了 MIND magnitude registration，MP-PCA 阶段再把去噪结果写回 `Mag_crop`。该链路没有对行、列进行索引裁剪。

因此，准确描述应为：NeSVoR 使用了由 **MIND 配准 + MP-PCA 后的 `Mag_crop` 变量**拟合并导出的 stack；这里的 `crop` 是历史变量名，**不是空间裁剪后的图像**。之后 Figure 1/2 使用的 ROI/display crop 是重建后的可视化步骤，未参与 NeSVoR 训练。

训练输入的本地来源（上传清单与服务器 `result.json` 一一对应）：

```text
fitting_qc/NIfTI_Slices/{sax,2ch,4ch}_{T1,T2}_stack_r0100.nii.gz
```

服务器训练路径为 `/data/dengyz/dataset/CYJ_mapping_v5/` 下同名文件。`full_nii` 在该实验结果中是另存副本，不能据此把本次训练定义为“未 crop 版本”；源码才是空间裁剪结论的依据。

## 正式比较的两组运行

双版本评估入口指定的正式比较对为：`nesvor_v5` 和 `nesvor_v5_128_w1`。名称含义也与实际参数一致：后一组使用 128 inference samples 和 pixel-variance weight 1.0。

| 参数 | `nesvor_v5` | `nesvor_v5_128_w1` |
| --- | ---: | ---: |
| `n_inference_samples` | 512 | 128 |
| `pixel_variance_weight` | 0.98 | 1.0 |

除输出目录/日志及上表两项外，两个 `result.json` 的重建配置相同。`nesvor_v5_128` 和 `nesvor_v5_256` 目录也存在，但不在正式双版本评估命令的 `--versions` 中，故不是这里所问的对比组合。

## 两组共用的必要 NeSVoR 参数

```yaml
input_stacks:
  T1: [sax_T1_stack_r0100.nii.gz, 2ch_T1_stack_r0100.nii.gz, 4ch_T1_stack_r0100.nii.gz]
  T2: [sax_T2_stack_r0100.nii.gz, 2ch_T2_stack_r0100.nii.gz, 4ch_T2_stack_r0100.nii.gz]
thicknesses_mm: [8, 8, 8]
normalization_before_reconstruction:
  T1_data_max_ms: 2500
  T2_data_max_ms: 200
output_resolution_mm: 1.0
output_psf_factor: 1.0
registration: stack
scanner_space: true
metric: none
weight_transformation: 1.0
weight_deform: 0.1
weight_image: 1.0
weight_pixel_var_smooth: 0.0
no_slice_scale: true
single_precision: true
log2_hashmap_size: 19
batch_size: 2000
n_iter: 10000
output_intensity_mean: 0.0
seed: 20260823
device: cuda:0

# NeSVoR model / optimization defaults recorded in result.json
coarsest_resolution: 16
finest_resolution: 0.5
level_scale: 1.3819
n_features_per_level: 2
depth: 1
width: 64
n_features_z: 15
n_features_slice: 16
n_samples: 256
inference_batch_size: 16000
no_pixel_variance: false
no_slice_variance: false
deformable: false
weight_bias: 100
image_regularization: edge
delta: 0.2
learning_rate: 0.005
gamma: 0.33
milestones: [0.5, 0.75, 0.9]
```

本实验的输出是 normalized map；反归一化应使用上述 T1/T2 data maximum（对应本地 `T1_norm_params.json` 与 `T2_norm_params.json`）。

## 共用的旧版 2D mapping 输入参数

以下是生成上述 NeSVoR 输入 map 的 **历史 v5 2D-fit-first** 物理/采集参数，不是当前 Code_10w/Trad 的协议参数：

```yaml
FA_deg: [45, 45, 45]
TI_ms: [10, 100]
T2_prep_ms: [35, 45, 55]
nr_startups: 10
duration_formula: delta_acquisition_time - TR*(EchoTrainLength+10)
preprocessing: MIND magnitude registration, then MP-PCA
```

## 可追溯证据

- 正式比较对：`/home/universe/SVR/multimap_postprogramming/MultiMapCode/method_repositories/2D_fit_first/EVALUATION_PROGRAMS_USAGE_ORDER.md`
- 上传的六个训练输入：`/home/universe/SVR/multimap_postprogramming/MultiMapCode/subjects/CYJ/results/2D_fit_first/experiments/20260822_v5_full_pipeline_v1/SERVER_UPLOAD_READY.md`
- 两组实际 T1/T2 训练参数：
  - `/home/universe/SVR/multimap_postprogramming/MultiMapCode/subjects/CYJ/results/2D_fit_first/experiments/20260822_v5_full_pipeline_v1/output/nesvor_v5/{T1,T2}/result.json`
  - `/home/universe/SVR/multimap_postprogramming/MultiMapCode/subjects/CYJ/results/2D_fit_first/experiments/20260822_v5_full_pipeline_v1/output/nesvor_v5_128_w1/{T1,T2}/result.json`
- 训练输入目录、旧 v5 mapping 参数：`/home/universe/SVR/multimap_postprogramming/MultiMapCode/subjects/CYJ/results/2D_fit_first/experiments/20260822_v5_full_pipeline_v1/config.yaml`
- `Mag_crop` 无空间裁剪、MIND 注册：`/home/universe/SVR/multimap_postprogramming/MultiMapCode/method_repositories/2D_fit_first/matlab/preprocessing/PreData.m`
- `Mag_crop` 的 MP-PCA 写回：`/home/universe/SVR/multimap_postprogramming/MultiMapCode/method_repositories/2D_fit_first/matlab/preprocessing/Run_MPPCA_All_Stacks.m`
- MAT-to-NIfTI export：`/home/universe/SVR/multimap_postprogramming/MultiMapCode/method_repositories/2D_fit_first/mapping_mat_to_slice_nifti_qc.py`
