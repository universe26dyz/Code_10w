# Final pre-experiment gate report

1. **Real VPS87 validation — FAIL.** The no-pool validator CLI is implemented in
   `validate_real_timings.py`, but no eligible trained checkpoint plus local
   CYJ/DYZ/HHZ/HJL prepared inputs were run together in this task.
2. **Lazy RR HDF5 loader — PASS.** `LazyRRSplit` opens HDF5 lazily, uses row
   reads, `num_workers=0`, and training `drop_last=True`; a tiny CPU smoke
   completed generation, training, and gradient diagnostics.
3. **RR decoder benchmark — FAIL.** Existing benchmark remains legacy-pool
   based and has not yet been refactored for RR dataset/checkpoint metadata.
4. **Dictionary cache — FAIL.** The MATLAB cache/refactor and equivalence test
   are not implemented.
5. **Common map-domain PSF — FAIL.** The exported-NIfTI common operator and
   deterministic geometry test are not implemented.

No expensive Trad or formal MLP job was run.
