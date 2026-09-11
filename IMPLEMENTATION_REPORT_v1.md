# MultiMap 10w v1 实现报告

更新：2026-09-11。以下明确区分工程可运行与科学验证。

## 最终目录与数据流

`trad/` 是冻结的直接 HHZ 基线；`mlp/` 包含 offline simulator training 和 online
reconstruction。两者共享 MATLAB preprocessing、bridge、10-weight group dataset、NeSVoR
HashGrid Quantitative INR、group-shared rigid、local anisotropic PSF、staged objective、
physical-RAS export 与 QC。数据流：DICOM 十 weights → MIND（`Mag_crop=MIND_mag_reg`）
→ prepared NPZ → 每 native slice 一个 complete group → INR T1/T2/B1/A → rigid/PSF →
10-weight balanced loss → NIfTI。两种方法在线仅 signal decoder 不同。

## 协议、vendor 与 Trad

- NeSVoR `master` `2e96a91bdd30174210caea911e03a2778c65adbe`；mDM `master`
  `a24ab1008d48e92ddf6f2bb8131f9a58dda23e95`。third_party 均为普通源码，无 `.git`。
- HHZ v1：NumImg=10、FA=[45,45,45]°、TI=[50,150] ms、T2prep=[35,45,55] ms、
  nRampUp=10，TR/VPS 来自 DICOM。DYZ 旧 TI=[10,100] ms 未被继承。
- Trad 是 HHZ recurrence 的直接 tensorized PyTorch 实现，无 EPG/deform/low-rank。此前
  20 个 deterministic MATLAB raw-signal parity 最大绝对误差为
  `1.2351231148954867e-15`。

## MLP 实现

offline MLP teacher 是 HHZ-compatible Trad simulator。输入
`[T1/1000,T2/1000,B1,timing2..10/1000]`，结构严格
`12 → (Linear 200 + BN + LeakyReLU)×3 → Linear 10`，输出 L2-normalized 10-HB
fingerprint。HDF5 split 一次读入 CPU-RAM；pool/dataset/checkpoint 记录并校验 SHA256、
protocol、timing min/max 与 functional-fixture 标志。

online `FrozenMLPSignalDecoder` 校验 checkpoint architecture/normalization/ranges/protocol/
TR/VPS/timing domain 后，直接注入既有 `TradQuantitativeForward`。参数冻结且不进
optimizer；BN 在外层 `train()` 下仍 eval，T1/T2/B1 input gradients 保留。

## 实际 tiny case、测试与 benchmark

实际 smoke 是 `CYJ_20260819_163756 / 2ch`：一个 group、十个 weights；MLP online 使用
Stage A=10、Stage B=4、PSF K=2。输出位于
`/tmp/multimap_phase6_mlp_online_smoke/outputs`：四个 NIfTI 均 finite、14 条有限日志、
rigid tensor 存在、无 deform。

执行并通过：针对性 offline regression（RAM HDF5、SHA mismatch、fixture guard）；
online suite：

```bash
conda run -n knesvr_torch pytest -q tests/test_mlp_checkpoint_compatibility.py \
  tests/test_mlp_online_decoder_contract.py tests/test_mlp_online_batchnorm_frozen.py \
  tests/test_mlp_end_to_end_tiny.py
# 5 passed
bash scripts/run_smoke_cpu.sh
```

Phase 5 tiny offline (1024/256/256、2 epochs) overall RMSE `0.0945053`、MAE
`0.0725863`、max error `0.2966225`。仅 decoder、同 input/device/dtype、3 次中位数的 CPU
benchmark：

| Batch | Trad forward/backward s | MLP forward/backward s | RMSE |
|---:|---:|---:|---:|
| 1,000 | 0.015359 / 0.068325 | 0.019783 / 0.005341 | 0.093809 |
| 10,000 | 0.069739 / 0.299684 | 0.025209 / 0.058581 | 0.093742 |

benchmark JSON：`/tmp/multimap_phase6_mlp_online_smoke/decoder_benchmark.json`。不计
HDF5/DICOM/INR/PSF/checkpoint loading。GPU benchmark launcher 已生成但本地未运行。

## 科学边界与服务器下一步

当前 tiny checkpoint 是 `functional_fixture=true`、`scientific_checkpoint=false`。因此工程上
可以说 Trad v1 end-to-end runnable、MLP v1 online integration runnable，且二者只更换
decoder；不能说 MLP 已达到 Trad accuracy、跨受试泛化或科学等效。

尚未运行：GPU training/benchmark、真实多受试 timing pool、formal MLP、formal 3-stack
reconstruction、Phase-6 MATLAB parity 与 image-quality comparison。服务器第一步应建立真实
SAX/2CH/4CH timing pool、训练 formal MLP、做 held-out timing fidelity，再执行：

```bash
cd /home/universe/SVR/multimap_postprogramming/Code_10w/mlp
PREPARED_SAX_NPZ=/absolute/sax/observations.npz \
PREPARED_2CH_NPZ=/absolute/2ch/observations.npz \
PREPARED_4CH_NPZ=/absolute/4ch/observations.npz \
MLP_CHECKPOINT=/absolute/formal/signal_simulator_best.pth \
OUTPUT_DIR=/absolute/output \
bash scripts/run_server_example.sh
```

仍需在正式数据上确认的科学问题：真实 timing-domain 覆盖、held-out fingerprint/derivative
fidelity、正式 decoder speed/memory、以及 Trad/MLP full reconstruction quantitative comparison。
