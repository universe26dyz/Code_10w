# Module 07 — Trad objective and staged training

基于 vendored NeSVoR `inr/train.py` 的 AdamW、encoding/network 参数分组、
MultiStepLR、中心化与空间缩放逻辑。`training_space.py` 保留 Module 02/03 的
physical RAS-mm truth，不原地修改 dataset；训练空间是 `(RAS-center)/30`。

默认训练只使用 one-subject robust trimmed-mean normalized intensity 的 10-weight
balanced MSE 与固定 inverse-frequency stack 权重；该 scalar 联合所有 masked
SAX/2CH/4CH samples 计算，不改变 relative intensities 或 dataset.v。Stage A 冻结 group rigid；Stage B 在同一模型上
开放 rigid 并按 NeSVoR `trans_loss` 相对 `axisangle_init` 正则。没有 bias、
slice/weight scale、low-rank 或 deform。

空间正则按 T1/T2/B1 分别配置 `none`、TV、L2 或 edge-preserving，并始终在
normalized parameter range 上计算。amplitude-guided 项是另行记录、另行加权的
T1/T2-only ablation，amplitude gradient 会 detach，B1 不使用该 prior。

`variance.enabled=false` 是默认值。开启时的
`experimental_project_heteroscedastic` 分支是本项目的 xyz/slice log-variance
head，具有独立 `training.learning_rates.variance`，不是 upstream NeSVoR
`sigma_net` 的等价实现。
