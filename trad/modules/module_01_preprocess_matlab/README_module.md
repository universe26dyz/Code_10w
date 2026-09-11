# Module 01 — MATLAB preprocessing (HHZ v1)

输入为一个原始 DICOM stack，输出为一个 MAT 文件。入口为
`preprocessing_v1/preprocess_stack_v1.m`；它不扫描受试者目录、不猜测
路径、不覆盖既有输出。调用方须显式传入 DICOM 输入目录、尚不存在的输出
MAT 路径与完整 `opts`。

```matlab
opts = preprocess_options_v1(2, false, false); % 仅 2 个 spatial slices 的 smoke
preprocess_stack_v1('/path/to/stack', '/path/to/output.mat', opts);
```

数据流严格为：按 `AcquisitionTime` 排序 → `[Nx,Ny,10,Nslice]` reshape →
可选 MP-PCA → MIND 将 HB2–HB10 配准到 HB1 → `Mag_crop = MIND_mag_reg` →
保存。每组 10 个 weight 在 MIND 后共享 HB1 几何；MAT 中保存了
`Info_by_slice` 和 `HB1_source_file_per_slice` 以供后续 bridge 使用。

`hhz_original/` 是未修改的 HHZ MATLAB 源码副本，不能作为 v1 入口。
`preprocessing_v1/` 只做路径化、显示开关、smoke slice 限制、可选 MP-PCA、
metadata 保存及 MIND 数据流修正。协议固定为 NumImg=10、FA=[45,45,45]、
TI=[50,150] ms、T2prep=[35,45,55] ms；TR/VPS 从 DICOM 的
`RepetitionTime`/`EchoTrainLength` 读取。DYZ 旧值 TI=[10,100] 仅作差异记录，
不得继承。
