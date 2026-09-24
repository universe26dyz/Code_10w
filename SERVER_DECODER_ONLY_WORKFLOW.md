# Server decoder-only workflow

This is the active server runbook.  It supersedes the archived subject/timing-pool
MLP runbooks.  Offline MLP training uses only the fixed HHZ/VPS=87 RR synthetic
dataset.  Online reconstruction is subject-specific and uses all valid SAX,
2CH and 4CH observations; Trad and FrozenMLP invoke the same reconstruction
engine and differ only at the signal-decoder factory.

The configured CYJ reference bundles are:

- native reference: `/data/dengyz/dataset/Code_10w_v1/reference_data/CYJ/native_reference_2d`
- myocardium masks: `/data/dengyz/dataset/Code_10w_v1/reference_data/CYJ/myocardium_masks_v2`

## 1. Define and verify inputs

Run this in the server shell.  Change only `CODE10W_ROOT` if the server checkout
is not `/data/dengyz/code/Code_10w`; all other paths are derived from the
currently configured dataset layout.

```bash
set -euo pipefail
export CODE10W_ROOT=/data/dengyz/code/Code_10w
export CODE10W_ENV=cr_dreme
export DATA_ROOT=/data/dengyz/dataset/Code_10w_v1
export PREPROCESSED_ROOT="$DATA_ROOT/Code_10w_preprocessed"
export PREPARED_ROOT="$DATA_ROOT/Code_10w_prepared"
export RUNS_ROOT="$DATA_ROOT/Code_10w_runs/decoder_only_v1"
export SUBJECT_ID=CYJ
export NATIVE_REFERENCE_ROOT="$DATA_ROOT/reference_data/CYJ/native_reference_2d"
export MYOCARDIUM_MASK_ROOT="$DATA_ROOT/reference_data/CYJ/myocardium_masks_v2"
cd "$CODE10W_ROOT"

test -f config/reference_paths.yaml
test -f "$NATIVE_REFERENCE_ROOT/native_reference_manifest.json"
test -f "$MYOCARDIUM_MASK_ROOT/manifest.json"
test -f "$MYOCARDIUM_MASK_ROOT/sax_myocardium_masks.npz"
conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m scripts.check_server_environment --mode formal-mlp \
  --output "$RUNS_ROOT/preflight/formal_mlp_environment.json"
conda run --no-capture-output -n "$CODE10W_ENV" python -c \
  "from trad.evaluation.scmr.reference_paths import load_reference_paths; p=load_reference_paths('config/reference_paths.yaml'); assert not p.skip_metrics, p.warnings; print(p)"
```

`CODE10W_ROOT` is the existing server checkout.  `PREPROCESSED_ROOT` is the
existing directory containing `<subject>/<stack>/preprocessed.mat`.
`CODE10W_ENV=cr_dreme` is the current repository-configured server Conda
environment (`configs/deployment_paths.example.yaml`).  If the server uses a
different environment name, change only this variable; do not change any
scientific YAML.
`PREPARED_ROOT` is the existing or new prepared-observation root.
`RUNS_ROOT` is a new experiment root: each child output below must be absent or
empty.  `SUBJECT_ID=CYJ` is required for the supplied CYJ reference bundles;
use a different subject only without this reference evaluation.  Do not use
`DYL` for FrozenMLP reconstruction: its VPS=85 is incompatible with the fixed
VPS=87 checkpoint.

Before GPU reconstruction, also run:

```bash
conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m scripts.check_server_environment --mode reconstruction \
  --output "$RUNS_ROOT/preflight/reconstruction_environment.json"
```

## 2. Prepare or verify the three-stack observations

If `$PREPARED_ROOT/$SUBJECT_ID/{sax,2ch,4ch}/observations.npz` already exists,
do not re-run preparation; verify it instead:

```bash
for stack in sax 2ch 4ch; do
  test -f "$PREPARED_ROOT/$SUBJECT_ID/$stack/observations.npz"
  test -f "$PREPARED_ROOT/$SUBJECT_ID/$stack/manifest.json"
  test -f "$PREPARED_ROOT/$SUBJECT_ID/$stack/qc_summary.json"
done
```

Otherwise create a new/empty `$PREPARED_ROOT` with the checked-in CYJ manifest:

```bash
conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m trad.scripts.prepare_all_observations \
  --manifest configs/subject_stack_manifest_CYJ_done.json \
  --preprocessed-root "$PREPROCESSED_ROOT" \
  --prepared-root "$PREPARED_ROOT"
```

The manifest is an existing input that supplies the server DICOM paths.  The
preprocessed root must already exist.  The prepared root is a new output only
when preparation has not been run; the command refuses to overwrite its batch
QC report.

## 3. Offline RR synthetic MLP, validation and approval

These commands never read subject observations.  `RR_DATASET_DIR` and
`MLP_CANDIDATE_DIR` are new/empty output directories.  The candidate checkpoint
is produced by the training command at
`$MLP_CANDIDATE_DIR/signal_simulator_best.pth`.

```bash
export RR_DATASET_DIR="$RUNS_ROOT/mlp_offline/rr_synthetic_20260923"
export MLP_CANDIDATE_DIR="$RUNS_ROOT/mlp_offline/candidate_20260923"
export MLP_CANDIDATE="$MLP_CANDIDATE_DIR/signal_simulator_best.pth"
export MLP_VALIDATION="$MLP_CANDIDATE_DIR/formal_validation.json"
export MLP_APPROVED="$RUNS_ROOT/mlp_offline/approved/signal_simulator_approved.pth"

conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m mlp.modules.module_05_signal_decoder.generate_rr_mlp_dataset \
  --config mlp/configs/rr_synthetic/formal_signalonly.yaml \
  --output-dir "$RR_DATASET_DIR"

conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m mlp.modules.module_05_signal_decoder.train_mlp \
  --config mlp/configs/rr_synthetic/formal_signalonly.yaml \
  --dataset-dir "$RR_DATASET_DIR" \
  --protocol mlp/configs/protocol_hhz_v1.yaml \
  --output-dir "$MLP_CANDIDATE_DIR"

conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m mlp.modules.module_05_signal_decoder.validate_formal_mlp \
  --checkpoint "$MLP_CANDIDATE" --dataset-dir "$RR_DATASET_DIR" \
  --protocol mlp/configs/protocol_hhz_v1.yaml --output "$MLP_VALIDATION" \
  --device cuda:0 --batch-size 1024 --gradient-samples 64
```

`RR_DATASET_DIR` is a new synthetic-data output.  `MLP_CANDIDATE_DIR` is a new
training output.  `MLP_CANDIDATE` is an existing file only after training.
`MLP_VALIDATION` is a new validation-report file.  The validation command does
not approve a checkpoint.

Formal RR training intentionally loads the `input12` and `target_signal10`
tensors into CPU RAM (about 1.1 GB across the formal train/valid/test splits).
This follows the upstream mDM-style training pattern and avoids random
row-wise HDF5 I/O.  It writes `train_history.csv` and
`signal_simulator_last.pth` after every completed epoch.  To resume an
interrupted run, keep the same dataset, config, and output directory, then run:

```bash
conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m mlp.modules.module_05_signal_decoder.train_mlp \
  --config mlp/configs/rr_synthetic/formal_signalonly.yaml \
  --dataset-dir "$RR_DATASET_DIR" \
  --protocol mlp/configs/protocol_hhz_v1.yaml \
  --output-dir "$MLP_CANDIDATE_DIR" \
  --resume "$MLP_CANDIDATE_DIR/signal_simulator_last.pth"
```

Resume verifies the run's history, scientific contract, and
`dataset_metadata.json` SHA256 before restoring model, optimizer, scheduler,
and RNG state. It appends only subsequent epochs to the existing history.

Before approval, run a real VPS=87 timing/fidelity audit.  The checked-in full
manifest contains `CYJ`, `DYZ`, `HHZ`, and `HJL` plus DYL; DYL is deliberately
excluded here.  This command reads prepared observations only for timing-domain
coverage and signal-fidelity audit: it never fits the MLP or alters the
candidate.  It fails if a group is not exactly the fixed protocol (VPS=87,
TR=2.61 ms) or has inconsistent within-group TR/VPS.  Confirm the four
prepared subject roots exist first; if one is unavailable, supply only verified
VPS=87 subject IDs and record that limitation in human review.

```bash
export REAL_TIMING_VALIDATION="$MLP_CANDIDATE_DIR/real_timing_validation.json"
conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m mlp.modules.module_05_signal_decoder.validate_real_timings \
  --checkpoint "$MLP_CANDIDATE" --prepared-root "$PREPARED_ROOT" \
  --subjects CYJ DYZ HHZ HJL --rr-dataset-dir "$RR_DATASET_DIR" \
  --protocol mlp/configs/protocol_hhz_v1.yaml --output "$REAL_TIMING_VALIDATION" \
  --device cuda:0 --samples-per-timing 16
```

This creates the new `$REAL_TIMING_VALIDATION` and sibling
`real_timing_validation_per_source.csv`. The formal gate is:

```text
synthetic held-out validation + real acquisition timing/fidelity validation
                                  ↓ both pass
                               human review
                                  ↓ approval
```

The approval command programmatically requires the real report to be
`rr_real_vps87_validation/v1`, `PASS`,
`eligible_for_human_review`, and `awaiting_manual_review`, and requires its
candidate SHA256 to match `$MLP_CANDIDATE`. It therefore rejects
`OUTSIDE_TRAINING_DOMAIN` and `do_not_approve` reports rather than treating
them as documentation-only advice.

After an authorized human has reviewed both reports, create a separate approved
checkpoint. The source candidate remains unchanged. Replace the review note
with the actual reviewer/date, not a placeholder.

```bash
conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m mlp.scripts.approve_formal_checkpoint \
  --checkpoint "$MLP_CANDIDATE" --validation-report "$MLP_VALIDATION" \
  --real-timing-validation "$REAL_TIMING_VALIDATION" \
  --approved-output "$MLP_APPROVED" \
  --review-note "Reviewed on YYYY-MM-DD by NAME; validation accepted."
```

`MLP_APPROVED` is a new output file.  Use it, not the candidate, for online
reconstruction.  Its `validation_status` must be `approved_by_manual_review`;
the FrozenMLP loader verifies that state, active RR dataset schema/split,
HHZ/VPS=87 protocol, architecture, normalization, parameter ranges and the
strict training timing domain before optimization starts.

## 4. Decoder-only reconstructions and timing profiles

Both output paths are new/empty directories.  Both commands consume the same
prepared root and subject and the B6-derived configuration; the MLP command
adds only the approved decoder checkpoint.

```bash
export TRAD_RUN="$RUNS_ROOT/$SUBJECT_ID/trad_B6"
export MLP_RUN="$RUNS_ROOT/$SUBJECT_ID/frozen_mlp_B6"

conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m trad.scripts.reconstruct_subject \
  --prepared-root "$PREPARED_ROOT" --subject-id "$SUBJECT_ID" \
  --config trad/configs/experiments_6k/B6.yaml --output "$TRAD_RUN"

conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m mlp.scripts.reconstruct_subject \
  --prepared-root "$PREPARED_ROOT" --subject-id "$SUBJECT_ID" \
  --config mlp_surrogate_v1/CYJ_B6/reconstruction.yaml --output "$MLP_RUN" \
  --mlp-checkpoint "$MLP_APPROVED"

test -f "$TRAD_RUN/timing_profile.csv"
test -f "$TRAD_RUN/timing_profile_summary.json"
test -f "$MLP_RUN/timing_profile.csv"
test -f "$MLP_RUN/timing_profile_summary.json"
```

The standard reconstruction outputs provide `T1_3D.nii.gz`, `T2_3D.nii.gz`,
and `final_rigid_poses.json` in each run directory.  Their summaries have the
same schema and explicitly declare `decoder_type: Bloch` or `FrozenMLP`.

## 5. Native-reference and myocardium evaluation, then paired comparison

This evaluation is valid for `CYJ` because the configured reference bundles are
CYJ bundles.  Each evaluation output is a new/empty directory.  The map-domain
operator uses the same PSF convention for the two completed reconstructions;
it validates the native reference provenance and requires a PASS myocardium
bundle before emitting myocardium metrics.

```bash
export TRAD_EVAL="$TRAD_RUN/evaluation_map_domain_psf"
export MLP_EVAL="$MLP_RUN/evaluation_map_domain_psf"
export COMPARISON="$RUNS_ROOT/$SUBJECT_ID/trad_vs_frozen_mlp_comparison.json"

conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m trad.evaluation.scmr.run_map_domain_psf_comparison \
  --method shared_svr --decoder-type Bloch --run-label trad_B6 --subject-id "$SUBJECT_ID" \
  --t1-volume "$TRAD_RUN/T1_3D.nii.gz" --t2-volume "$TRAD_RUN/T2_3D.nii.gz" \
  --prepared-root "$PREPARED_ROOT" --final-poses "$TRAD_RUN/final_rigid_poses.json" \
  --native-reference "$NATIVE_REFERENCE_ROOT" --preprocessed-root "$PREPROCESSED_ROOT" \
  --mask-bundle "$MYOCARDIUM_MASK_ROOT" --n-samples 32 --seed 20260921 --output "$TRAD_EVAL"

conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m trad.evaluation.scmr.run_map_domain_psf_comparison \
  --method shared_svr --decoder-type FrozenMLP --run-label frozen_mlp_B6 --subject-id "$SUBJECT_ID" \
  --t1-volume "$MLP_RUN/T1_3D.nii.gz" --t2-volume "$MLP_RUN/T2_3D.nii.gz" \
  --prepared-root "$PREPARED_ROOT" --final-poses "$MLP_RUN/final_rigid_poses.json" \
  --native-reference "$NATIVE_REFERENCE_ROOT" --preprocessed-root "$PREPROCESSED_ROOT" \
  --mask-bundle "$MYOCARDIUM_MASK_ROOT" --n-samples 32 --seed 20260921 --output "$MLP_EVAL"

conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m trad.evaluation.scmr.compare_decoder_runs \
  --trad-map-metrics "$TRAD_EVAL/map_domain_psf_metrics.json" \
  --mlp-map-metrics "$MLP_EVAL/map_domain_psf_metrics.json" \
  --trad-timing-summary "$TRAD_RUN/timing_profile_summary.json" \
  --mlp-timing-summary "$MLP_RUN/timing_profile_summary.json" \
  --output "$COMPARISON"
```

The comparison output is new and reports only paired numeric deltas as
FrozenMLP minus Bloch/Trad, plus run-level timing fields.  It does not change
either reconstruction or infer scientific superiority from metric signs.

## 6. Decoder benchmark

Run this after approval.  It is a separate decoder microbenchmark, not a
reconstruction result.  `BENCHMARK_OUTPUT` is a new JSON file and
`RR_DATASET_DIR` is the existing output from section 3.

```bash
export BENCHMARK_OUTPUT="$RUNS_ROOT/mlp_offline/decoder_benchmark.json"
conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m mlp.modules.module_09_qc_benchmark.benchmark_signal_decoder \
  --checkpoint "$MLP_APPROVED" --rr-dataset-dir "$RR_DATASET_DIR" \
  --protocol mlp/configs/protocol_hhz_v1.yaml --output "$BENCHMARK_OUTPUT" \
  --device cuda:0 --batch-size 2560 --batch-size 5120 --batch-size 10240 --repetitions 3
```
