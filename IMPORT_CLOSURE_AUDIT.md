# Import Closure Audit

## Scope and method

Static scan performed on 2026-09-22 from the repository root:

```bash
rg -n "^(from|import) (modules|third_party)(\\.|\\s|$)" -g '*.py'
```

The scan found 151 ambiguous top-level import statements across `trad/` and
`mlp/`.  It intentionally records all matches before changing only the formal
Trad reconstruction runtime closure.  No experimental data, result folders,
or checkpoint payloads are touched by this migration.

## Runtime-critical Trad closure

The following direct dependency chain is required to import and execute the
formal Trad reconstruction entry point.  These are the only production files
migrated in Phase B.

| Runtime role | Source file | Legacy root(s) | Phase B action |
| --- | --- | --- | --- |
| CLI entry | `trad/scripts/run_training.py` | `modules` | `trad.modules` |
| Subject entry | `trad/scripts/reconstruct_subject.py` | `scripts` | `trad.scripts` |
| Trainer / optimizer / scheduler / loss | `trad/modules/module_07_objective_training/trad_trainer.py` | `modules` | `trad.modules` |
| Training coordinates | `trad/modules/module_07_objective_training/training_space.py` | `modules`, `third_party` | `trad.modules`, `trad.third_party` |
| Dataset / normalization | `trad/modules/module_03_dataset_geometry/quantitative_point_dataset.py` | `modules`, `third_party` | `trad.modules`, `trad.third_party` |
| INR | `trad/modules/module_04_quantitative_inr/quantitative_inr.py` | `third_party` | `trad.third_party` |
| Rigid PSF forward model | `trad/modules/module_06_rigid_psf/rigid_psf_forward.py` | `third_party` | `trad.third_party` |
| HB1 initialization | `trad/modules/module_06_rigid_psf/hb1_stack_adapter.py` | `modules`, `third_party` | `trad.modules`, `trad.third_party` |
| Geometry bridge | `trad/modules/module_02_data_bridge/geometry.py` | `third_party` | `trad.third_party` |
| Formal export | `trad/modules/module_08_inference_export/export_quantitative.py` | `modules`, `third_party` | `trad.modules`, `trad.third_party` |

`export_quantitative.py` is included because `run_training.py` calls it during
the same formal reconstruction invocation; the export computations themselves
are unchanged.  The signal decoder remains the existing Trad decoder at this
phase.  Therefore this import migration does not change the decoder-only
scientific comparison or checkpoint state-dict keys.

## Matches deliberately outside Phase B

### Trad

- Evaluation: `trad/evaluation/scmr/run_psf_k8_gpu_stage.py`.
- Legacy/support scripts: `trad/scripts/export_checkpoint.py`,
  `trad/scripts/prepare_all_observations.py`, and
  `trad/scripts/validate_transferred_preprocessed.py`.
- Tests: all current `trad/tests/test_*.py` matches retain their legacy path
  shim until the shared reconstruction core migration.  They do not determine
  production import behavior.

### MLP

All MLP trainer, data, PSF, export, script, benchmark, and test matches remain
unchanged in this phase.  The only MLP module imported by the new isolation
test is the package-qualified frozen decoder, whose internal imports are
already relative.  Its formal trainer will move only when it is replaced by
the shared reconstruction orchestration core.

## Full scan inventory by directory

| Directory | Matches | Classification |
| --- | ---: | --- |
| `trad/modules` | 19 | all 19 are runtime-critical statements in the Phase B closure |
| `trad/scripts` | 12 | five statements in `run_training.py` are Phase B; other scripts are deferred |
| `trad/evaluation` | 5 | deferred evaluation |
| `trad/tests` | 34 | deferred test-only legacy imports |
| `mlp/modules` | 21 | deferred pending shared-core migration |
| `mlp/scripts` | 12 | deferred pending shared-core migration |
| `mlp/tests` | 48 | deferred test-only / MLP-specific imports |

The count includes both `from modules...` and `from third_party...` matches.
Phase B validates the closed package-qualified Trad runtime path in a fresh
interpreter, then imports the package-qualified frozen MLP decoder in that
same interpreter.  Passing means neither `modules` nor `third_party` can be
registered as a top-level namespace during the tested import closure.
