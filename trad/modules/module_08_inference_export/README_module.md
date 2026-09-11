# Module 08 — physical-RAS inference and export

训练完成后，volume query 只走一个可逆边界：physical RAS-mm
`→ (RAS-center_ras_mm)/spatial_scaling → QuantitativeINR`。输出 grid 和 NIfTI
affine 均明确为 RAS-mm，写出 `T1_3D.nii.gz`、`T2_3D.nii.gz`、`B1_3D.nii.gz`、
`amplitude_3D.nii.gz` 和 undo centering/scaling 后的 `final_rigid_poses.json`。
