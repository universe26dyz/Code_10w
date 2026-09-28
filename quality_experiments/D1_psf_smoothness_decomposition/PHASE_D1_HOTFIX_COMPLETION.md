# D1 central-plane loader hotfix completion report

- Status: `CODE_READY_SERVER_RUN_PENDING`
- Root cause: the v1 loader incorrectly required one central-plane archive row per spatial group. The baseline exporter writes one row for every acquired observation; each group has exactly ten rows with weights `0..9`.
- Reference verified: `trad.evaluation.scmr.run_scmr_fig12._extract_quantitative_stack` uses the correct historical interpretation.
- Exact fix: require `weight_idx`; validate the complete zero-based group set, exactly ten rows and exactly one of each weight per group; select the unique weight-0 T1/T2 row; AND all ten observation masks; retain the exact native array shape and reject any mismatch.
- Formal result: NOT RUN / NOT AVAILABLE. The first CMRServer04 attempt failed before metrics with `FAILED_PRECOMPUTE_INPUT_SCHEMA_VALIDATION`; `baseline_v1` is preserved and is not a formal result.
- Corrected server result root: `/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/quality_experiments_v1/CYJ/D1_psf_smoothness_decomposition_baseline_v2/`.
- Local code/test environment: explicit `knesvr_torch` CPU.
- Validation: compileall passed; `pytest -q quality_experiments/D1_psf_smoothness_decomposition/test_d1_metrics.py` passed (13 tests); runner `--help` passed; `bash -n quality_experiments/server_commands/run_D1_CYJ.sh` passed.
- Implementation commit: `43b0424` — `Fix D1 central-plane observation grouping`.
- Files changed by implementation commit: `io.py`, runner manifest revision, observation-level regression tests, v2 revision policy, ledger, quality policy README, phase state, and D1 server script.
- Git status after implementation commit: clean before this completion-record update.
- PUSH PERFORMED: NO.
- Next manual action: user manually pushes the local commit, server pulls it, then user runs `quality_experiments/server_commands/run_D1_CYJ.sh`.
