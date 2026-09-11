# Module 04 — Quantitative INR

`QuantitativeINR` consumes RAS-mm coordinates and uses NeSVoR's vendored
`HashEmbedder`, `build_encoding`, `build_network`, and
`compute_resolution_nlevel`.  A shared latent feeds four physical fields:
`T2=5+195*sigmoid`, `T1=T2+(2500-T2)*sigmoid`,
`B1=0.1+1.1*sigmoid`, and `A=softplus+eps`.

It has no positional encoding, deformation, bias, variance, or slice-scale
parameters.

CPU 时按 NeSVoR 原生 `USE_TORCH=True` 使用 vendored PyTorch HashGrid；CUDA 且
tinycudann 可用时保留 NeSVoR 的 `build_encoding` backend 选择，不以具体类型
阻断 tcnn。正式训练传入 `spatial_scaling=30`，以训练坐标和 physical-mm hash
resolution 保持 NeSVoR 的计算关系。
