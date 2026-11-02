# 08 Evaluation and visualisation

All metrics, AHA maps, slice-pairs and QC use outputs from module 07. LV is `outer_filled` (the hand-drawn circle plus its interior). Myocardium keeps both `myocardium_core_1px` and `myocardium_core_legacy`; blood pool uses the legacy endocardial fill. Report all-slice, held-out-only and per-stack strata with voxel count, mean, SD, RMSE, MAE, NMAE, bias, CC/NCC, PSNR and zero audit.

`run_stage.sh` first runs common-native metrics/QC and then all-stack ROI metrics for the selected case.

Use the two server command wrappers shown in the Chinese manual, with explicit `SUBJECT`, `REPO`, `PY` and ROI paths. A shape mismatch is a pose/plane error; never hide it by cropping arrays.
