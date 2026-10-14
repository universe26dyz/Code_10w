# INVALID FOR SCIENTIFIC USE

This route combined Background XCAT anatomy with `Patient1/Mesh.vtk` myocardium although they are not in the same geometry/deformation state.

Measured evidence: centroid distance is approximately 18.50 mm and raw-world Dice is approximately 0.455.

Affected historical output: `paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_1_clean_mrxcat`.

The code is retained only for audit/provenance. Do not import or call it from active experiment code. Its replacement is the Phase-1.2 `xcat3d` route.
