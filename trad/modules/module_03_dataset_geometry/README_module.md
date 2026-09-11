# Module 03 — 10-weight group dataset and geometry

`QuantPointDataset` 复用 vendored NeSVoR `PointDataset` 的核心思路：将 masked
pixels 展平为 untransformed `xyz`、intensity `v` 和批次采样。必要差异是它保留
`group_idx`、`weight_idx`、`stack_idx` 和 9D timing，而不把每个 weight 设为
独立 pose。

```python
from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
dataset = QuantPointDataset('/path/to/prepared_case/observations.npz')
```

严格契约：group id 必须连续且每组恰有 weight 0..9；一组中所有 weights 共用
同一 HB1 cropped affine 与 timing；`group_pose_count` 因而等于 native spatial
slice 数，供下一阶段建立“一组一个 future rigid pose”。`xyz` 当前保持明确的
DICOM-LPS millimetres；任何进入 NeSVoR 的坐标转换必须在后续边界显式完成。
