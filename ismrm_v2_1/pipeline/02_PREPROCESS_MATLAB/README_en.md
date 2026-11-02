# 02 MATLAB preprocessing

Run MATLAB on the server to convert the ten-weight DICOM groups into `preprocessed.mat` and `preprocess_qc.json` for SAX, 2CH and 4CH. This is the pre-fitting input stage; never reuse observations produced with the wrong TI timing.

```bash
bash "$PROJECT_ROOT/common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/server_preprocess_and_prepare.sh" \
  "$RUN_ROOT" "$SUBJECT" "$PROJECT_ROOT/subjects/$SUBJECT/manifest.json" \
  "$PROJECT_ROOT/configs/protocol_scmr_v5_10_100.yaml" matlab "$PYTHON"
```

Verify ten channels, MP-PCA enabled and `center_data=true`. The launcher refuses to overwrite existing preprocessed or prepared roots.

