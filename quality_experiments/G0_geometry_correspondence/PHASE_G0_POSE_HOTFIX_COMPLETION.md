# G0 pose-audit hotfix completion report

- Status: `CODE_READY_SERVER_RUN_PENDING`
- Root cause: G0 v1 treated NeSVoR axis-angle vectors as `[translation, rotation]`. Vendored `RigidTransform` uses `[rotation-vector, translation]` with `trans_first=True`.
- G0 v1 preserved result: correspondence/Hungarian findings remain valid (all six formal combinations were diagonal, without non-identity orientation). Only pose-derived output fields are invalid.
- Exact fix: pose audit now uses vendored `RigidTransform(initial).inv().compose(final).axisangle(trans_first=True)` for relative deltas and `ax_transform_points` for final local-origin centers and translation-free normals.
- Implementation commit: `df336eb` — `Fix G0 pose audit transform semantics`.
- Files changed by implementation commit: `pose_audit.py`, G0 manifest revision, transform-reference regression tests, v2 result-revision policy, v2 server target, ledger, and phase state.
- Validation: G0 compileall passed; G0 targeted pytest passed (12 tests); direct runner help passed using CPU transform fallback; G0 server-script syntax check passed.
- Local formal rerun: not possible; exact CYJ inputs remain server-only under `/data/...`.
- Formal v2 command: `quality_experiments/server_commands/run_G0_CYJ.sh`.
- Formal v2 output root: `/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/quality_experiments_v1/CYJ/G0_geometry_correspondence_baseline_v2/`.
- Git status after implementation commit: two unrelated untracked files, `et --hard HEAD@{0}` and `git reset --hard HEAD@{1}`, were present and deliberately not staged, changed, or deleted.
- PUSH PERFORMED: NO.
- Next manual action: user manually pushes the local commits, server pulls them, then user runs the corrected G0 v2 server script. Stop after this hotfix; do not begin D2.
