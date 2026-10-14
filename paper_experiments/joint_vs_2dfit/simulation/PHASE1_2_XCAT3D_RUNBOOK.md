# Phase-1.2 XCAT3D runbook

This route requires a licensed external XCAT runtime. Codex runs only local CPU validation and must not generate the XCAT files.

1. Create the hand-off template locally:

   `python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage prepare-xcat --output-root /path/to/xcat3d_handoff`

2. In the licensed XCAT environment, use one unchanged XCAT configuration, resolution, cardiac phase, and respiratory phase to generate full-3-D T1, T2, signal-support (or PD/M0), and LV-myocardium volumes. Set the LV values in the XCAT parameterization to T1=1410 ms and T2=50 ms. Do not use Patient1 mesh or any 2-D VTI output.

3. Complete `xcat3d_inputs.template.json` with filenames, raw-binary shape/dtype/axis order, world geometry, and hashes of the shared XCAT parameter files. Copy it beside the four generated files as `xcat3d_inputs.json`.

4. Validate before creating a dataset:

   `python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage validate-xcat-inputs --xcat-input-manifest /path/to/xcat3d_inputs.json`

5. Build the local GT and then render only after the gate passes:

   `python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage build-gt --xcat-input-manifest /path/to/xcat3d_inputs.json --output-root paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_2_xcat3d_clean`

   `python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage simulate-signal-field --output-root paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_2_xcat3d_clean --prepared-root /path/to/Code_10w_prepared --approved-decoder paper_experiments/joint_vs_2dfit/audit/outputs/approved_decoder.json`

   `python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage render-observations --output-root paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_2_xcat3d_clean --prepared-root /path/to/Code_10w_prepared`

No formal reconstruction is part of this runbook.
