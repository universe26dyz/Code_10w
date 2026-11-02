# 06 Proposed

Proposed consumes sparse observations, the corrected `[10,100]` decoder, B6, protocol and the sparse manifest. It does not consume fitting maps. L100 uses `$RUN_ROOT/prepared/$SUBJECT`; every other case uses its sparse case root. The output root must be fresh.

`run_stage.sh` runs one case at a time and checks the manifest and fresh output root before launching.

Pass all expected SHA256 values explicitly. Each completed case must contain T1/T2 NIfTI, an experiment manifest and run provenance. A pre-existing partial output is an error, not a reason to overwrite.
