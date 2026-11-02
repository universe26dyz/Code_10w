# 07 Final pose

Project both completed methods into one acquisition-native final pose. This stage does not retrain or reconstruct. It consumes completed volumes, fit-first native maps, prepared geometry and the subject ROI bundle.

`run_stage.sh` invokes the paired final-pose bundle for one completed case.

```bash
SUBJECT="$SUBJECT" REPO="$PROJECT_ROOT/common" PY="$PYTHON" \
bash "$PROJECT_ROOT/common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_paired_final_pose.sh" \
  "$RUN_ROOT" "$RUN_ROOT" "$PROJECT_ROOT/subjects/$SUBJECT/roi" L100_fold0
```

Continue only after native-cache and prepared-SAX provenance checks pass.
