# MultiMap 10w MLP v1

MLP v1 与 Trad v1 共用 prepared observation、10-weight group dataset、NeSVoR
HashGrid Quantitative INR、T1/T2/B1/A ranges、group-shared rigid pose、local
anisotropic PSF、balanced MSE、regularization、AdamW stages 和输出 sampler。在线唯一
替换为 `TradSignalSimulator → FrozenMLPSignalDecoder`；没有 EPG、deform、low-rank、
bias、variance、per-weight/slice scale。

HHZ 是 v1 权威：NumImg=10、FA=[45,45,45]°、TI=[50,150] ms、T2prep=[35,45,55]
ms、nRampUp=10；TR/VPS 来自 DICOM `RepetitionTime`/`EchoTrainLength`。DYZ 的
TI=[10,100] ms 不会继承。MIND 后 `Mag_crop=MIND_mag_reg`，十个 weights 共用 HB1
geometry 与一个 rigid pose。

## Offline signal-simulator training

teacher 是 HHZ-compatible Trad simulator，不是 EPG、dictionary 或 inverse model。输入为
`[T1/1000,T2/1000,B1,timing2..10/1000]`；网络严格为
`12 → (Linear(200), BN, LeakyReLU) × 3 → Linear(10)`，再 L2 normalize 为 10-HB
fingerprint。T1∈[20,2500] ms、T2∈[5,200] ms、B1∈[0.1,1.2]，且 T1>T2。

`build_timing_pool.py` 要求每 group 完整十 weights、相同 TR（容差）和 VPS（精确）；
HDF5 按 unique timing 切分。训练将每个 split 一次读入 CPU-RAM TensorDataset，training
batch≥2 且 drop-last。`teacher_device` 必须在离线 config 显式指定。pool SHA256、protocol、
timing min/max、functional-fixture 状态写入 dataset metadata、checkpoint 和 resolved config。

```bash
cd /home/universe/SVR/multimap_postprogramming/Code_10w/mlp
conda run -n knesvr_torch python -m modules.module_05_signal_decoder.build_timing_pool \
  --observations /absolute/prepared/observations.npz --output /absolute/timing_pool.npz
conda run -n knesvr_torch python -m modules.module_05_signal_decoder.generate_mlp_dataset \
  --config configs/mlp_server_train_example.yaml --timing-pool /absolute/timing_pool.npz \
  --protocol configs/protocol_hhz_v1.yaml --output-dir /absolute/synthetic_h5
conda run -n knesvr_torch python -m modules.module_05_signal_decoder.train_mlp \
  --config configs/mlp_server_train_example.yaml --dataset-dir /absolute/synthetic_h5 \
  --timing-pool /absolute/timing_pool.npz --protocol configs/protocol_hhz_v1.yaml \
  --output-dir /absolute/mlp_signal_model
```

## Online quantitative SVR reconstruction

checkpoint loader 强制 architecture、normalization、parameter ranges、完整 HHZ protocol、
TR/VPS 和 online timing domain 兼容。formal config 的
`allow_functional_fixture_checkpoint=false` 会拒绝 functional checkpoint。冻结 decoder
无 `torch.no_grad()`，不进 optimizer；外层 `train()` 后 BN 仍为 eval 且运行统计必须不变。

本地功能 smoke（不是科学重建评价）：

```bash
cd /home/universe/SVR/multimap_postprogramming/Code_10w/mlp
bash scripts/run_smoke_cpu.sh
```

服务器三 stack joint reconstruction：

```bash
cd /home/universe/SVR/multimap_postprogramming/Code_10w/mlp
PREPARED_SAX_NPZ=/absolute/sax/observations.npz \
PREPARED_2CH_NPZ=/absolute/2ch/observations.npz \
PREPARED_4CH_NPZ=/absolute/4ch/observations.npz \
MLP_CHECKPOINT=/absolute/formal/signal_simulator_best.pth \
OUTPUT_DIR=/absolute/output \
bash scripts/run_server_example.sh
```

GPU benchmark launcher 是 `scripts/benchmark_signal_decoder_gpu.sh`，本地未运行。只有完成
真实跨受试 timing pool、formal MLP training、held-out fidelity、正式 benchmark 与完整
reconstruction comparison 后，才可讨论 MLP quantitative accuracy 或 cross-subject
generalization。
