# Formal MLP scientific validation v1

状态：**readiness 代码完成，awaiting formal server execution / manual review**。

## 固定科学定义

MLP teacher 为 HHZ-compatible TradSignalSimulator；网络固定
`12 → (Linear 200 + BN + LeakyReLU)×3 → Linear 10`，输入为
`[T1/1000,T2/1000,B1,timing2..10/1000]`，输出为 normalized 10-HB fingerprint。
不使用 EPG、deformable、low-rank、bias/variance，也不修改 Trad frozen physics。

## formal 输入与 split

必须提供 JSON manifest，显式列出每个 `subject_id`、`stack` 与 prepared NPZ；不得从路径猜
subject。正式 pool 保留每 group provenance（即使 timing 数值重复）。要求严格 subject-disjoint
split：排序后固定 3 train / 1 valid / 1 test，且每 subject 的 SAX/2CH/4CH 位于同一 split。

```bash
cd /home/universe/SVR/multimap_postprogramming/Code_10w/mlp
conda run -n knesvr_torch python -m modules.module_05_signal_decoder.build_timing_pool \
  --manifest /absolute/formal_sources.json \
  --output /absolute/formal_mlp_v1/timing/timing_pool.npz
conda run -n knesvr_torch python -m modules.module_05_signal_decoder.formal_timing_audit \
  --pool /absolute/formal_mlp_v1/timing/timing_pool.npz \
  --csv /absolute/formal_mlp_v1/timing/formal_timing_audit.csv \
  --summary /absolute/formal_mlp_v1/timing/formal_timing_audit_summary.json
```

只有 audit 显示 TR 在容差内一致且 VPS 完全相同才可继续。否则停止，不扩展 12D 输入。

## 一次 formal candidate

配置必须显式填入实际排序 subject 的 3/1/1 split，并固定 seed `20260911`、CUDA teacher、
train/valid/test `1,000,000/100,000/100,000`、50 epochs、batch 1024、Adam 1e-3、
StepLR(5,0.5)。只运行一次 generation 与一次训练：

```bash
conda run -n knesvr_torch python -m modules.module_05_signal_decoder.generate_mlp_dataset \
  --config /absolute/formal_config.yaml --timing-pool /absolute/formal_mlp_v1/timing/timing_pool.npz \
  --protocol configs/protocol_hhz_v1.yaml --output-dir /absolute/formal_mlp_v1/synthetic
conda run -n knesvr_torch python -m modules.module_05_signal_decoder.train_mlp \
  --config /absolute/formal_config.yaml --dataset-dir /absolute/formal_mlp_v1/synthetic \
  --timing-pool /absolute/formal_mlp_v1/timing/timing_pool.npz \
  --protocol configs/protocol_hhz_v1.yaml --output-dir /absolute/formal_mlp_v1/model
conda run -n knesvr_torch python -m modules.module_05_signal_decoder.validate_formal_mlp \
  --checkpoint /absolute/formal_mlp_v1/model/signal_simulator_best.pth \
  --dataset-dir /absolute/formal_mlp_v1/synthetic --timing-pool /absolute/formal_mlp_v1/timing/timing_pool.npz \
  --protocol configs/protocol_hhz_v1.yaml --device cuda:0 --batch-size 1024 --gradient-samples 64 \
  --output /absolute/formal_mlp_v1/validation/formal_validation_report.json
```

训练 checkpoint 仅为 `formal_candidate=true, validation_status=unvalidated`；validation report
写 `awaiting_manual_review`，不会自动批准 online reconstruction。报告必须检查 held-out test
对 train timing domain 是 interpolation 还是 extrapolation。

## GPU decoder benchmark

formal candidate 经人工审查批准后，才以 1,000/10,000/100,000、one warm-up、3 次中位数运行
decoder-only CUDA benchmark。100,000 OOM 要原样记录，不可缩小后冒充。Trad/MLP peak VRAM
必须分别 reset/measure；不把 CPU RSS 解释为方法显存比较。

## 当前实际结果

本地 Phase 7 未运行 audit、synthetic generation、formal training、formal validation 或 GPU
benchmark：当前执行机没有 `nvidia-smi`/CUDA，且未提供 explicit multi-subject prepared manifest。
因此没有可报告的 formal subject split、runtime、best epoch、fidelity、GPU speed 或 GPU memory。
也没有开始任何 full Trad/MLP 3-stack reconstruction。
