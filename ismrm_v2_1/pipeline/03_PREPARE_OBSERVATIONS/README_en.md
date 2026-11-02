# 03 Prepare observations

This bridge reads the MAT outputs, DICOM identities and the SCMR-v5 10/100 protocol and writes `observations.npz` plus per-stack manifests. The NPZ is generated here, not supplied as raw input.

`run_stage.sh` invokes `prepare_all_observations.py` with explicit manifest and protocol paths.

```bash
python "$PROJECT_ROOT/common/trad/scripts/prepare_all_observations.py" \
  --manifest "$PROJECT_ROOT/subjects/$SUBJECT/manifest.json" \
  --preprocessed-root "$RUN_ROOT/preprocessed" \
  --prepared-root "$RUN_ROOT/prepared" \
  --protocol "$PROJECT_ROOT/configs/protocol_scmr_v5_10_100.yaml"
```

Manifests must bind TR/VPS/TI/T2-prep, ten weight indices and source SHA256. Treat any source-observation mismatch as a provenance error.
