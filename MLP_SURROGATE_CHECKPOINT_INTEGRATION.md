# FrozenMLP checkpoint integration and server runbook

## Non-negotiable gate

No formal MLP reconstruction may start until the selected frozen-decoder
checkpoint passes the existing `load_frozen_mlp_decoder` gate.  For a formal
candidate this requires all existing metadata checks, including matching HHZ
protocol and timing domain, plus:

```text
validation_status: approved_by_manual_review
```

Do **not** change `validation_status` manually（不修改 validation metadata）.  A fixture checkpoint remains
permitted only for CPU tests when `allow_functional_fixture_checkpoint: true`;
the CYJ_B6 template sets it to `false`.

## Server preparation

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

## Gate-only dry check

This command is safe before a formal run: it proves the package entry and
requires the explicit decoder argument, but does not reconstruct anything.

```bash
cd <CODE10W_ROOT>
conda run --no-capture-output -n knesvr_torch \
  python -m mlp.scripts.reconstruct_subject --help
```

The following command is the formal run command.  Execute it only after the
approval state above is present and the paths have been checked by the operator:

```bash
cd <CODE10W_ROOT>
conda run --no-capture-output -n knesvr_torch \
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
