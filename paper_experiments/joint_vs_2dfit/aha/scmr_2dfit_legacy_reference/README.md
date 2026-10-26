# SCMR / 2D-fit-first source snapshot

This directory is an exact, read-only source snapshot of the historical SCMR/
2D-fit-first ROI, AHA16, myocardial-quantification, and visualization modules.
It is stored inside the current `Code_10w` tree so that the CYJ1.1 evaluation
workflow can be archived as one self-contained codebase.

## Provenance

The files were copied without edits from:

`MultiMapCode/method_repositories/2D_fit_first/python/`

`SHA256SUMS` records the copied bytes.  Verify the snapshot from this directory:

```bash
sha256sum -c SHA256SUMS
```

## Relationship to the current CYJ1.1 runner

The executable CYJ1.1 runner remains entirely within the current `Code_10w`
tree.  It does **not** import from the original absolute source directory and
does not execute this snapshot directly.  The historical files require their
former package layout and the old evaluation/reconstruction context; executing
them as-is would reintroduce unrelated legacy paths and does not support the
paired proposed-versus-fit-first CYJ output contract.

The current implementation is a narrowly adapted, self-contained reuse of the
historical semantics:

- `evaluation/roi/build_myocardium_masks.py`: full-stack myocardial masks;
- `evaluation/roi/map_native_roi_to_final_pose.py`: exact native-plane ROI
  mapping;
- `evaluation/metrics/evaluate_myocardial_quantitative_fidelity.py`: per-slice
  myocardial statistics;
- `evaluation/aha16/`: AHA16 labels, statistics, and validation;
- `visualization/scmr/make_figure3_bullseye.py`: historical SCMR Figure 3
  layout.

The maintained entry point is
`paper_experiments.joint_vs_2dfit.aha.run_cyj11_aha_roi`; its current modules
are the only runtime dependency for CPU evaluation.  This snapshot is retained
for auditability and long-term preservation, not as a second executable
pipeline.
