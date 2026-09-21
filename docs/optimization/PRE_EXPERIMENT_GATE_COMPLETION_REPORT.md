# Pre-experiment gate completion report

| Gate | Status | Evidence |
| --- | --- | --- |
| Trad commands | PASS | `reconstruct_subject` resolves only sax/2ch/4ch; commands use `Code_10w_prepared`. |
| REG_FULL | PASS (hard blocked) | `experiment.blocked=true` is rejected before dataset creation. |
| RR generation → training | PASS (CPU smoke) | signal-only HDF5 generated and trained without a timing pool. |
| RR synthetic validation | PASS (CPU smoke) | on-the-fly exact-Bloch signal and gradient diagnostics ran. |
| Real VPS87 validation | FAIL | Utility is not yet implemented. |
| Decoder benchmark | FAIL | RR no-pool benchmark support is not yet implemented. |
| Dictionary cache | FAIL | MATLAB cache/equivalence implementation is not yet present. |
| Common map-domain PSF | FAIL | exported-volume common operator is not yet present. |
| Tests | PASS (targeted) | Trad path/block/config tests and RR schema tests pass. |

No 6k Trad reconstruction, pilot/formal MLP job, formal dataset generation, or multi-subject reconstruction was run.
