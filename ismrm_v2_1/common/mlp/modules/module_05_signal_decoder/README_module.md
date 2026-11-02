# Module 05B — offline HHZ Trad-teacher MLP

Teacher 是本目录 `trad_teacher/TradSignalSimulator` 的 HHZ-compatible PyTorch
副本，不是 EPG。唯一 active dataset 是固定 HHZ/VPS=87 的 RR synthetic
rhythm split；它不读取 subject prepared NPZ，也不使用 subject/timing-pool split。

输入固定为 `[T1/1000,T2/1000,B1,timing9/1000]`，输出是 L2-normalized 10-HB
fingerprint。`MdmSignalMLP` 是 vendored mDM 的 12→200→200→200→10 最小 input
变体：三组 Linear/BatchNorm/LeakyReLU 和最后 Linear。HDF5 dataset 仅生成一次，
按 rhythm 切分 train/valid/test；每 epoch 不重跑 teacher。

`FrozenMLPSignalDecoder` 固定 parameters 与 BatchNorm 为 eval，但 forward 不用
`torch.no_grad()`，因此可将 gradient 回传到未来在线 INR 的 T1/T2/B1 输入。
在线 reconstruction 通过 shared reconstruction engine 注入 frozen decoder；唯一变量是 decoder。
