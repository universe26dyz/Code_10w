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

唯一正式离线流程为 RR synthetic：固定 HHZ protocol（TR=2.61 ms、VPS=87）生成
rhythm-disjoint train/valid/test HDF5；不读取真实 subject observation，也不做 subject
split。训练逐行读取 HDF5，training batch≥2 且 drop-last。`teacher_device` 必须在离线
config 显式指定；protocol、rhythm split 与 timing-domain 写入 dataset metadata、checkpoint
和 resolved config。

```bash
cd /home/universe/SVR/multimap_postprogramming/Code_10w
conda run --no-capture-output -n knesvr_torch \
  python -m mlp.modules.module_05_signal_decoder.generate_rr_mlp_dataset \
  --config mlp/configs/rr_synthetic/formal_signalonly.yaml \
  --output-dir /absolute/new_rr_synthetic_dataset
conda run --no-capture-output -n knesvr_torch \
  python -m mlp.modules.module_05_signal_decoder.train_mlp \
  --config mlp/configs/rr_synthetic/formal_signalonly.yaml \
  --dataset-dir /absolute/new_rr_synthetic_dataset \
  --protocol mlp/configs/protocol_hhz_v1.yaml \
  --output-dir /absolute/new_mlp_candidate
```

## Online quantitative SVR reconstruction

checkpoint loader 强制 architecture、normalization、parameter ranges、完整 HHZ protocol、
TR/VPS 和 online timing domain 兼容。formal config 的
`allow_functional_fixture_checkpoint=false` 会拒绝 functional checkpoint。冻结 decoder
无 `torch.no_grad()`，不进 optimizer；外层 `train()` 后 BN 仍为 eval 且运行统计必须不变。

服务器三 stack joint reconstruction 的唯一推荐入口：

```bash
cd /home/universe/SVR/multimap_postprogramming/Code_10w
conda run --no-capture-output -n knesvr_torch python -m mlp.scripts.reconstruct_subject \
  --prepared-root /absolute/prepared_root --subject-id CYJ \
  --config mlp_surrogate_v1/CYJ_B6/reconstruction.yaml \
  --output /absolute/new_mlp_run --mlp-checkpoint /absolute/approved_checkpoint.pth
```

GPU benchmark 使用 `python -m mlp.modules.module_09_qc_benchmark.benchmark_signal_decoder`
并且只接受 RR synthetic dataset。只有完成 RR held-out validation、人工 approval、正式
benchmark 与完整 reconstruction comparison 后，才可讨论 MLP quantitative accuracy。
