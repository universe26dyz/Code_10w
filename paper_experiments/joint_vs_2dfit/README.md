# Joint vs. 2D-fit-first paper experiments

This namespace contains only new paper-specific contracts, audit tools, and
future experiment code. Phase 0 is read-only with respect to prepared data and
historical runs; it does not run reconstruction, simulation, noise, motion, or
sparsity experiments.

Run each audit with `--help`. Audit outputs refuse to overwrite an existing
file. The approved decoder must be resolved from an explicit checkpoint or B6
provenance, never selected by filename.
