# DYL_2 end-to-end command plan

The formal route is conditional on the input audit becoming `PASS` after the MATLAB/bridge timing checks. Every output path below must be new; restart means choose a new run tag after diagnosing the prior immutable result.

| stage | location / runtime | input -> output | expected provenance |
|---|---|---|---|
| DICOM audit | LOCAL CPU / light | DICOM -> audit report | source counts, UIDs, timing evidence |
| MATLAB preprocess | LOCAL CPU / moderate | DICOM -> `preprocessed.mat` | subject/stack, protocol checks, crop geometry |
| Bridge | LOCAL CPU / moderate | MAT + DICOM -> prepared observations | UID, affine, timing, TR/VPS manifests |
| Transfer / validation | SERVER_CPU / moderate | prepared -> immutable sparse manifests | checksums, `condition` + `phase` |
| R0/L100 | SERVER_GPU / expensive | full prepared -> R0 outputs/cache | frozen dependencies and route manifest |
| sparse L50/L25 | SERVER_GPU / expensive | sparse prepared -> method outputs | held-out original group identities |
| Q002/Q004 | SERVER_GPU / expensive | prepared + frozen B6/decoder -> endpoints | model/input/dependency SHA256, seed, endpoint |
| Fit-first | SERVER_CPU+GPU / expensive | dictionary maps -> independent T1/T2 NeSVoR | normalization, denormalization, separate final poses |
| endpoint bundles/metrics | SERVER_CPU / moderate | endpoints + cache/mask -> metrics | endpoint, mask and pose provenance |
| figures/summary | SERVER_CPU / moderate | metrics -> figures/summary | formal case IDs and fold aggregation |

Use `docs/CURRENT_FORMAL_PIPELINE_COMMANDS.md` for exact command templates. The only sparse grid is L100/fold0, L50/fold0..1, L25/fold0..3. L100 uses final learned/optimized pose as method-specific; L50/L25 sample the 3-D endpoints on original held-out native geometry without fabricated/nearest/retained pose.
