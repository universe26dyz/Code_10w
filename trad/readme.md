# MultiMap 10w Trad（Phase 1 scaffold）

本目录将实现直接定量 SVR：10 个 MIND 配准后的 2D weighted observations
最终由定量 INR、HHZ 信号模型与 NeSVoR rigid/PSF 共同解释。与传统
`2D dictionary matching → NeSVoR` 不同，目标链路是：

```text
DICOM 10 weights → MATLAB MIND preprocessing → Mag_crop
→（后续）quantitative INR →（后续）HHZ signal decoder
→（后续）NeSVoR rigid + PSF → 2D data consistency → 3D T1/T2
```

Phase 2 已实现 MATLAB v7.3→Python bridge 与 10-weight group dataset；INR、
signal decoder、rigid/PSF、训练与导出仍未实现。

科学协议固定为 NumImg=10、FA=[45,45,45]°、TI=[50,150] ms、
T2prep=[35,45,55] ms、nRampUp=10，TR/VPS 分别来自 DICOM 的
`RepetitionTime`/`EchoTrainLength`。HHZ 是 v1 权威；DYZ 旧脚本曾使用
TI=[10,100] ms，v1 不会无声沿用。MIND 后 `Mag_crop` 必须等于
`MIND_mag_reg`，且一个 spatial group 的十个 weights 共用 HB1 几何。

`third_party/UPSTREAM.md` 记录 NeSVoR 来源；该目录不含 `.git`。v1 将禁用
EPG、NeSVoR deformable、low-rank、额外模型、per-weight scale、bias 和
variance（后续实现时的默认约束）。
