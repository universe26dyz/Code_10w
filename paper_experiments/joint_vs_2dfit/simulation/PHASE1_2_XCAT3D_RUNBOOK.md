# Phase-1.2 final gated XCAT3D workflow

Codex performs local CPU code checks only. Run licensed XCAT only in its licensed environment; do not run JEMRIS, GPU reconstruction, Phase-2, real-subject experiments, or MRXCAT/Patient1 mesh routes here. Hanson is the source of the full-3D activity-field mechanism, not the Code_10w phantom content.

Before final XCAT geometry, use the separate scout workflow: inspect local context, create a scout geometry, freeze its handoff only when an exact XCAT build is known, then run **only** `lv_mask_phase1_2.par` manually in licensed XCAT. Return the raw scout LV mask to `analyze-xcat-scout`, then review `propose-final-xcat-geometry`. Scout data is design/audit data only and never enters formal GT, signal, rendering, or finalization.

```bash
python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage inspect-xcat-scout-context --output-root /path/to/scout_context
python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage make-xcat-scout-geometry --xcat-version-build 'EXACT LICENSED BUILD' --output-root /path/to/scout_geometry
python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage analyze-xcat-scout --geometry-spec /path/to/cardiac_roi_geometry.scout_v1.json --lv-mask-raw /path/to/lv_mask.bin --dtype float32 --prepared-root /path/to/prepared_real_cohort --output-root /path/to/scout_analysis
python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage propose-final-xcat-geometry --geometry-spec /path/to/cardiac_roi_geometry.scout_v1.json --scout-qc /path/to/xcat_scout_lv_qc.json --output-root /path/to/final_geometry_proposal
```

1. Inspect `xcat3d/cardiac_roi_phantom_spec.json`.
2. Copy `xcat3d/cardiac_roi_geometry.template.json` and fill a placeholder-free cardiac geometry, including XCAT build compatibility, matrix, slice range, static inputs, cm-to-mm convention, layout policy, centered synthetic-LPS origin policy, and PCA threshold.
3. Freeze the formal handoff locally:

   `python -m paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d --stage freeze-xcat-handoff --geometry-spec /path/to/cardiac_roi_geometry.json --output-root /path/to/handoff_parent`

4. Review all four generated `.par` files and `parameter_contract_qc.json`; no placeholder is executable.
5. Record parameter/spec/static-input path, SHA256, and role provenance in the formal `xcat3d_inputs.json`.
6. In licensed XCAT run the audit-only structure-ID field where supported, then T1, T2, PD, and LV-mask with the exact same frozen anatomy/geometry block.
7. Record raw file SHA256 values. The formal manifest derives `zyx` shape from `startslice..endslice` and `array_size`, derives spacing from cm × 10, and derives centered synthetic origin; do not enter an arbitrary raw origin.
8. Validate parameter semantics, static-input hashes, raw unit/layout, quantitative tuple coverage, and structure audit status.
9. Canonicalize to 1-mm synthetic xyz/LPS, validate LV crop margin and PCA stability, then build GT.
10. Jointly select one real subject for SAX and the two orthogonal real-geometry-derived LAX templates (`2ch` and `4ch`; neither is claimed as a landmark-certified clinical cut), and apply one global rigid transform.
11. Check every exact 6×6×9 midpoint quadrature node is in bounds; rendering is blocked otherwise.
12. Simulate HHZ, render SAX/2ch/4ch observations, and independently revalidate signal and observation semantics.
13. Write the immutable final manifest. It hashes all acceptance artifacts and cross-checks all final raw/spec/parameter/static provenance against the hashed source manifest.
14. Only a complete `phase1_2_final_manifest.json` permits Phase-2 preflight.
