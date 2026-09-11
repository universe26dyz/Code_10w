# Module 03 — 10-weight group dataset and geometry

`QuantPointDataset` 复用 vendored NeSVoR `PointDataset` 的核心思路：将 masked
pixels 展平为 untransformed `xyz`、intensity `v` 和批次采样。必要差异是它保留
`group_idx`、`weight_idx`、`stack_idx` 和 9D timing，而不把每个 weight 设为
独立 pose。

```python
from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
dataset = QuantPointDataset(['/path/to/prepared_case/observations.npz'])
```

严格契约：group id 必须连续且每组恰有 weight 0..9；一组中所有 weights 共用
同一 HB1 cropped affine 与 timing；`group_pose_count` 因而等于 native spatial
slice 数，供下一阶段建立“一组一个 rigid pose”。`xyz` 是 centered local
`[column,row,slice]` mm，`group_resolution_xyz_mm` 是 `[col,row,thickness]`。
`initial_transformation` 从 cropped HB1 DICOM-LPS affine 显式转换到 RAS-mm；
`xyz_transformed` 与 bounding box 都使用该初始 group pose。多个 NPZ 输入会全局
重新编号 group，同时原样保留每个 stack id。
