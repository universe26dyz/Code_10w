# Phase-1.2 cardiac-ROI quantitative XCAT runbook

Codex performs only local CPU validation. Hanson is the reference for the quantitative-XCAT generation mechanism (multiple full-3D activity-field runs with a shared XCAT anatomy, phase, rasterizer and geometry). Code_10w uses its own cardiac-ROI phantom specification; it is not a reproduction of Hanson phantom content. Do not use MRXCAT VTI files or a Patient1 mesh.

1. Inspect and freeze `xcat3d/cardiac_roi_phantom_spec.json`.
2. Copy `xcat3d/cardiac_roi_geometry.template.json` to a placeholder-free licensed-XCAT geometry specification. Its matrix, `startslice` and `endslice` must be decided from the actual cardiac ROI.
3. Locally produce the four shared-geometry parameter files:

   `python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage prepare-xcat --output-root /path/to/handoff_parent`

4. In the licensed XCAT environment run `t1_phase1_2.par`, `t2_phase1_2.par`, `pd_phase1_2.par`, and `lv_mask_phase1_2.par` with that one frozen geometry. The LV-mask run is `myoLV=1` and every other declared cardiac-ROI structure `=0`.
5. Record raw float32 paths/SHA256/dtype/shape/axis-order/spacing/origin/convention, XCAT version/build, cardiac and respiratory phase, each parameter-file SHA256, and shared cardiac-phantom/geometry-spec SHA256 in `xcat3d_inputs.json`. Placeholders are rejected.
6. Validate the four raw runs, canonicalize them with nearest-neighbour tissue-map conversion to xyz/LPS/1-mm, and require the configured LV boundary margin:

   `python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage validate-xcat-inputs --xcat-input-manifest /path/to/xcat3d_inputs.json`

7. Build GT, derive LV centroid/PCA and one shared SAX/2CH/4CH rigid transform, simulate HHZ, and render explicitly with 6x6x9 signal-space integration.
8. Run `--stage validate-final`. It writes a final immutable manifest only after SHA256 closure over GT, QC, signal, timing, source manifest, all three observations, T1/T2/PD/LV-mask source hashes, and the cardiac specs.
9. Only a complete, untampered `phase1_2_final_manifest.json` permits Phase-2 preflight.

No formal reconstruction is included here.
