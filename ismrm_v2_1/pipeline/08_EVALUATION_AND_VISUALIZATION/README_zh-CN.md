# 08 评估与可视化

metrics、AHA、slice-pair 和 QC 必须使用 07 的 final-pose 输出。LV 使用 `outer_filled`（圆圈及内部），心肌同时保留 `myocardium_core_1px` 与 `myocardium_core_legacy`，血池使用 legacy 内膜填充。统计要区分 all-slice、held-out-only 和 stack，并输出 voxel count、mean、SD、RMSE、MAE、NMAE、bias、CC/NCC、PSNR 及 zero audit。

`run_stage.sh`：按 case 先做 common-native metrics/QC，再做 all-stack ROI metrics。

```bash
SUBJECT="$SUBJECT" REPO="$PROJECT_ROOT/common" PY="$PYTHON" \
bash "$PROJECT_ROOT/common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_metrics_qc.sh" \
  "$RUN_ROOT" "$RUN_ROOT" "$PROJECT_ROOT/subjects/$SUBJECT/roi" L100_fold0
OUTPUT_NAME=all_stack_roi_metrics_v2_blood_pool_zero_audit \
SUBJECT="$SUBJECT" REPO="$PROJECT_ROOT/common" PY="$PYTHON" \
bash "$PROJECT_ROOT/common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_all_stack_roi_metrics.sh" \
  "$RUN_ROOT" "$RUN_ROOT" "$PROJECT_ROOT/subjects/$SUBJECT/roi" L100_fold0
```

图像与 metrics 必须引用同一 final-pose mask；若 shape 不同，停止并检查 plane/pose，不要裁剪数组来掩盖错误。
