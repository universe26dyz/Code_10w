# Module 07 — Trad objective and staged training

基于 vendored NeSVoR `inr/train.py` 的 AdamW、encoding/network 参数分组、
MultiStepLR、中心化与空间缩放逻辑。`training_space.py` 保留 Module 02/03 的
physical RAS-mm truth，不原地修改 dataset；训练空间是 `(RAS-center)/30`。

训练只使用 one-subject robust trimmed-mean normalized intensity 的 10-weight
balanced MSE 与固定 inverse-frequency stack 权重；该 scalar 联合所有 masked
SAX/2CH/4CH samples 计算，不改变 relative intensities 或 dataset.v。Stage A 冻结 group rigid；Stage B 在同一模型上
开放 rigid 并按 NeSVoR `trans_loss` 相对 `axisangle_init` 正则。没有 bias、
variance、slice/weight scale、low-rank 或 deform。

空间 field regularization 是 epsilon=1e-12 稳定的 L2 norm。训练在 optimizer step
前后分别检查当前 trainable gradients/parameters 有限；intensity normalization
provenance 同时写入 resolved config 与 checkpoint，导出只对 amplitude 乘回 scale。
