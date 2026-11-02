# 07 Final pose

把 Proposed 与 fit-first 结果投影到同一个 acquisition-native final pose；不重新训练、不重新 reconstruction。输入是两种方法的完成 volume、fit-first native maps、prepared/source geometry 和 subject ROI bundle。

`run_stage.sh`：对一个已完成 case 调用 paired final-pose bundle 入口。

```bash
SUBJECT="$SUBJECT" REPO="$PROJECT_ROOT/common" PY="$PYTHON" \
bash "$PROJECT_ROOT/common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_paired_final_pose.sh" \
  "$RUN_ROOT" "$RUN_ROOT" "$PROJECT_ROOT/subjects/$SUBJECT/roi" L100_fold0
```

输出在 `$RUN_ROOT/evaluation/final_pose/$SUBJECT`。先检查 native-cache contract 和 prepared-SAX provenance PASS，再继续 metrics。
