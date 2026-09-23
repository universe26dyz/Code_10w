# FrozenMLP checkpoint integration and server runbook

## Non-negotiable gate

No formal MLP reconstruction may start until the selected frozen-decoder
checkpoint passes the existing `load_frozen_mlp_decoder` gate.  For a formal
candidate this requires all existing metadata checks, including matching HHZ
protocol, active `mlp_rr_synthetic/v1` / `rhythm` dataset metadata, and the
strict training timing domain (not the broader held-out timing range), plus:

```text
validation_status: approved_by_manual_review
```

Do **not** change `validation_status` manually（不修改 validation metadata）.  A fixture checkpoint remains
permitted only for CPU tests when `allow_functional_fixture_checkpoint: true`;
the CYJ_B6 template sets it to `false`.

## Server preparation

Set `CODE10W_ENV=cr_dreme`, the repository-configured server Conda environment.
If the server uses a different name, change only that shell variable.

1. Deploy the exact committed repository revision to the server.  Confirm the
   server checkout has both `reconstruction_core/` and
   `mlp_surrogate_v1/CYJ_B6/reconstruction.yaml`.
2. Locate an approved MLP checkpoint and keep its original file unchanged.
   Record its absolute path as `<APPROVED_MLP_CHECKPOINT>`.
3. Locate the prepared three-stack observation root and the actual subject ID;
   record them as `<PREPARED_ROOT>` and `<SUBJECT_ID>`.  The wrapper expects:
   `<PREPARED_ROOT>/<SUBJECT_ID>/{sax,2ch,4ch}/observations.npz`.
4. Confirm the target output directory is new or empty.  Use only
   `Code_10w_runs/mlp_surrogate_v1/CYJ_B6`; do not write into `trad_v3` or
   `mlp_rr_v3`.

## Existing validation and approval workflow

Run held-out RR synthetic validation and the separate real VPS=87 timing/fidelity
audit on the formal candidate, review both reports outside the code, then
perform the existing explicit approval transition. The approval command remains
bound cryptographically to the held-out report only; the real-timing report is
a mandatory human-review condition and never participates in model fitting.
Neither validation nor approval mutates the candidate checkpoint.

```bash
cd <CODE10W_ROOT>
export CODE10W_ENV=cr_dreme
conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m mlp.modules.module_05_signal_decoder.validate_formal_mlp \
  --checkpoint <FORMAL_CANDIDATE_CHECKPOINT> \
  --dataset-dir <RR_SYNTHETIC_DATASET_DIR> \
  --protocol mlp/configs/protocol_hhz_v1.yaml \
  --output <VALIDATION_REPORT_JSON> \
  --device cuda:0 --batch-size <BATCH_SIZE> --gradient-samples <N_GRADIENT_SAMPLES>

# Only after an authorised human review of the report:
conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m mlp.scripts.approve_formal_checkpoint \
  --checkpoint <FORMAL_CANDIDATE_CHECKPOINT> \
  --validation-report <VALIDATION_REPORT_JSON> \
  --approved-output <APPROVED_MLP_CHECKPOINT> \
  --review-note "<AUTHORSED_REVIEW_NOTE>"
```

The approval command accepts only an `unvalidated` non-functional formal
candidate and a matching `awaiting_manual_review` report, then writes a new
approved file with `approved_by_manual_review`.  It does not modify the source
checkpoint or existing validation metadata.

## Gate-only dry check

This command is safe before a formal run: it proves the package entry and
requires the explicit decoder argument, but does not reconstruct anything.

```bash
cd <CODE10W_ROOT>
conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m mlp.scripts.reconstruct_subject --help
```

The following command is the formal run command.  Execute it only after the
approval state above is present and the paths have been checked by the operator:

```bash
cd <CODE10W_ROOT>
conda run --no-capture-output -n "$CODE10W_ENV" \
  python -m mlp.scripts.reconstruct_subject \
  --prepared-root <PREPARED_ROOT> \
  --subject-id <SUBJECT_ID> \
  --config mlp_surrogate_v1/CYJ_B6/reconstruction.yaml \
  --output <RUNS_ROOT>/mlp_surrogate_v1/CYJ_B6 \
  --mlp-checkpoint <APPROVED_MLP_CHECKPOINT>
```

The config is blocked without `--mlp-checkpoint`.  Supplying it only unlocks
the template; it does not bypass loader validation.  Any mismatch aborts before
optimization.  A successful run writes the common `timing_profile.csv` and
`timing_profile_summary.json`, then the standard reconstruction/export outputs.

## Local CPU status

The current workstation has no CUDA device.  It has completed only source-level
and tiny CPU tests; it has not run this formal command and has not created or
modified any historical experiment result.
