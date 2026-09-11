# Module 02 — MATLAB/Python data bridge

入口是 `prepare_observations.py`。它只接受 Phase 1 生成的 MATLAB `-v7.3`
MAT、显式当前 `--dicom-dir` 和 `--stack`，使用唯一的 `h5py` loader 恢复
`Mag_crop[Nx,Ny,10,Nslice]` 与 `Acq_time[10,Nslice]`。不尝试 scipy 或其他
fallback loader。

```bash
conda run -n knesvr_torch python -m modules.module_02_data_bridge.prepare_observations \
  --preprocessed-mat /path/to/stack_v1.mat \
  --dicom-dir /path/to/stack \
  --stack sax --output-dir /path/to/prepared_case --max-groups 1
```

MAT 保存的 SOPInstanceUID 是唯一定位键；bridge 扫描 `--dicom-dir` 后建立
`UID → current path`，缺失或重复 UID 立即报错。一个 native spatial slice 是
一个 `group_idx`，且必须恰有 weight 0..9。每个 group 从 HB1 的完整 DICOM
geometry 构造 DICOM-LPS(mm) `affine_lps_rc`，再用 HHZ crop 的实际 zero-based
row/column offset 更新 origin。该 affine 映射 `[row,col,slice,1]`；10 个
MIND 后 weights 共用同一 cropped HB1 affine。

`timing.py` 逐项翻译 HHZ `function_T1T2_10HB_bssfp.m` 的
`Duration_befor_Acq`：输出 `[10]`，bridge 将其 `[1:10]` 保存为每个 group/observation
的 9D conditioning vector。输出为 `observations.npz`、`manifest.json`、
`timing.npy` 和 `qc_summary.json`，都写入调用者指定目录，不复制 DICOM。

`observations.npz` 同时逐 observation 保存由已验证 MAT `TR`/`VPS` 取得的
`tr_ms`/`vps`。这是唯一的正式训练 protocol provenance；桥接不猜测参数。
