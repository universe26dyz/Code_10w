# Joint vs. 2D-fit-first paper experiments

This namespace contains only new paper-specific contracts, audit tools, and
future experiment code. Phase 0 is read-only with respect to prepared data and
historical runs; it does not run reconstruction, simulation, noise, motion, or
sparsity experiments.

Phase-1.1 synthetic GT is retired from scientific use because it combined
Background XCAT anatomy with a Patient1 LV mesh from a mismatched geometry
state. The active replacement is Phase-1.2: a single-source full-3-D XCAT
quantitative phantom under `xcat3d/`. Historical code and output remain only
as recycle/archive provenance. Phase-1.2 has four distinct statuses:
`LOCAL_CODE_ACCEPTANCE`, `EXTERNAL_XCAT_INPUT_GATE`,
`SCIENTIFIC_GT_GEOMETRY_GATE`, and `FORMAL_SYNTHETIC_READY`; only the first is
locally verifiable without actual XCAT files.

Run each audit with `--help`. Audit outputs refuse to overwrite an existing
file. The approved decoder must be resolved from an explicit checkpoint or B6
provenance, never selected by filename.
