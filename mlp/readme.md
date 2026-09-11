# MultiMap 10w MLP（Phase 1 scaffold）

本目录将实现使用 frozen universal signal MLP 的定量 SVR。目标链路是：

```text
DICOM 10 weights → MATLAB MIND preprocessing → Mag_crop
→（后续）quantitative INR →（后续）frozen MLP signal decoder
→（后续）NeSVoR rigid + PSF → 2D data consistency → 3D T1/T2
```

MLP 将是跨受试者 universal simulator，不是 subject-specific model、不是
inverse T1/T2 estimator、不是 hard dictionary matching，online reconstruction
也不会生成 dictionary。其唯一 teacher 将是后续 HHZ-compatible Trad
simulator，而不是 EPG。

Phase 2 已实现 MATLAB v7.3→Python bridge 与 10-weight group dataset；
`module_05_signal_decoder/trad_teacher/` 仍只是预留目录，MLP、teacher、训练与
重建尚未实现。

科学协议固定为 NumImg=10、FA=[45,45,45]°、TI=[50,150] ms、
T2prep=[35,45,55] ms、nRampUp=10，TR/VPS 分别来自 DICOM 的
`RepetitionTime`/`EchoTrainLength`。HHZ 是 v1 权威；DYZ 旧脚本的
TI=[10,100] ms 不会被继承。MIND 后 `Mag_crop = MIND_mag_reg`；每个 spatial
group 的十个 weights 必须共用 HB1 几何。

`third_party/UPSTREAM.md` 记录 NeSVoR 与 mDM 参考来源。mDM 仅保留三个
MLP 参考 Python 文件，不复制 EPG。所有 `third_party` 内容均为普通文件，
不含 `.git`。后续 v1 仍将禁用 NeSVoR deformable、low-rank、额外模型、
per-weight scale、bias 与 variance。
