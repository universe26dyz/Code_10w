# Phase-1.2 Hanson-anchored XCAT3D runbook

Codex performs only local CPU validation. Run licensed XCAT externally; do not substitute MRXCAT VTI files or a Patient1 mesh.

1. Locally create the handoff package:

   `python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage prepare-xcat --output-root /path/to/handoff_parent`

2. In the licensed XCAT environment, copy the supplied `XCAT_input_ap.dat` and `XCAT_input_dia.dat` paths into the four parameter files. Run the same anatomy, resolution, cardiac phase and respiratory phase four times: `t1_phase1_2.par` → T1 activity volume, `t2_phase1_2.par` → T2 volume, `pd_phase1_2.par` → PD volume, and `lv_mask_phase1_2.par` → LV mask volume. This reproduces the Hanson activity-field mechanism; the LV run is the explicit Code_10w extension.

3. Record each raw float32 file’s SHA256, raw shape/order, physical convention, spacing, origin, XCAT version/build and parameter-file hashes in `xcat3d_inputs.json`. Do not leave placeholders.

4. Point the local manifest at those raw files. Code_10w canonicalizes every volume together to xyz/LPS/mm and a 1-mm isotropic grid, validates the shared geometry, builds GT, derives LV centroid/PCA, aligns all views with one rigid transform, simulates HHZ, and renders with 6×6×9 quadrature:

   `python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage validate-xcat-inputs --xcat-input-manifest /path/to/xcat3d_inputs.json`

   `python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage build-gt --xcat-input-manifest /path/to/xcat3d_inputs.json --output-root paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_2_xcat3d_clean`

   `python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage simulate-signal-field --output-root paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_2_xcat3d_clean --prepared-root /path/to/Code_10w_prepared --approved-decoder paper_experiments/joint_vs_2dfit/audit/outputs/approved_decoder.json`

   `python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage render-observations --output-root paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_2_xcat3d_clean --prepared-root /path/to/Code_10w_prepared`

   `python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage validate-final --output-root paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_2_xcat3d_clean`

Only a passing final manifest permits Phase-2 preflight. No formal reconstruction is included here.
