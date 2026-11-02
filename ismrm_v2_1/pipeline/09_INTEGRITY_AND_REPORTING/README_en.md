# 09 Integrity and reporting

This read-only final stage records source commit, environment versions, protocol/model SHA256, observation identities, per-subject/case completion and migration hashes, then writes summaries. Every rerun uses a new run root.

```bash
python "$PROJECT_ROOT/scripts/verify_project_integrity.py" "$PROJECT_ROOT"
python "$PROJECT_ROOT/scripts/verify_runtime_contract.py" --root "$PROJECT_ROOT"
python "$PROJECT_ROOT/scripts/verify_no_legacy_code_path.py" "$PROJECT_ROOT"
```

Never commit patient data, model binaries or result trees. Summary CSV/JSON files refer to manifest paths and SHA256 values.

