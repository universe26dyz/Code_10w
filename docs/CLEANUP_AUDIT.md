# Cleanup audit

No data or code was moved in this audit. This is intentional: the required proof of non-use was not available for any nontrivial candidate, while the repository already contains explicit `archive/` provenance and formal artifacts cite historical paths.

| classification | decision / evidence |
|---|---|
| KEEP_FORMAL | Q002/Q004/Fit-first sources, `trad/modules`, sparse/mask/bundle/metric/reporting code, frozen decoder/B6 references, R0 native cache, masks, pose audit, endpoint exports and final summaries. Their producers/consumers are in `docs/CURRENT_FORMAL_PIPELINE_INVENTORY.json`. |
| KEEP_PROVENANCE | `archive/`, committed audit outputs, simulation artifacts and historical manifests. For example, `phase1_1_clean_mrxcat` is referenced by committed JSON manifests and `archive/recycle_code/.../PROVENANCE.json`. |
| KEEP_RULE_REFERENCE | `trad/evaluation/scmr/` and legacy myocardium/SCMR rule code. Historical geometry-transplant is deprecated as a primary path but the rules remain referenced by current semantics. |
| REVIEW_REQUIRED | Local data under `/home/universe/SVR/data/Code_10w_v1/rubbish` and `Code_10w_runs/rubbish`. It is outside this Git commit and was not moved because a manifest/SHA/provenance reverse-reference proof was not completed. |
| REVIEW_REQUIRED | Current untracked workspace items: `2D-FIT-FIRST_PIPELINE.png`, `Codex_joint_vs_2Dfit_分步骤执行包/`, `Joint_vs_2Dfit_完整实验执行方案.md`, `et --hard HEAD@{0}`, `git reset --hard HEAD@{1}`, `paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_1_clean_mrxcat/matlab_parity_final.json`, and `对比实验新增设计计划_1018.md`. They predated this work and are left untouched/uncommitted. |

The required quarantine exists at `_garbage/20261021_pipeline_audit/` with an empty manifest, README, and no-op restore script. The safe server cleanup procedure is proposal-only: first generate an itemized checksum/consumer-reference manifest on the server, review it, then move a selected immutable-copy-safe item while preserving relative path. Never use `rm -rf`, `git clean -fdx`, or `git reset --hard`.
