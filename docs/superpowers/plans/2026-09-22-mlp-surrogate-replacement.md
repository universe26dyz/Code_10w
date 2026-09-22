# MLP Surrogate Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Trad reconstruction orchestration the sole training route and inject either the Bloch or frozen MLP signal decoder while producing comparable profiling artifacts.

**Architecture:** Refactor the existing Trad trainer to receive a decoder factory while retaining reconstruction construction, normalization, loss, optimizer, staged-loop, and export behavior. The MLP route becomes an adapter that constructs only the existing validated frozen decoder. Extend current profiler records and summary instead of creating a timing system.

**Tech Stack:** Python, PyTorch, PyYAML, existing Trad/MLP modules, pytest.

**Spec:** `docs/superpowers/specs/2026-09-22-mlp-surrogate-replacement-design.md`

## Global Constraints

- Signal decoder is the sole Bloch-versus-MLP experimental variable.
- No formal reconstruction may run until the existing MLP validation gate accepts an approved checkpoint.
- Do not alter checkpoint validation metadata or bypass `load_frozen_mlp_decoder`.
- Do not hard-code an MLP checkpoint in the new CYJ_B6 configuration.
- Retain the existing CSV columns: `iteration,stage,section,milliseconds`.
- Do not modify historical result directories or generate substitute evaluation data.

## Review Focus

- Candidate checkpoints that are functional fixtures, unvalidated, protocol-mismatched, or timing-out-of-domain remain rejected; Task 2 runs the existing loader compatibility tests.
- Bloch and MLP resolved controlled settings differ only in decoder fields; Task 2 compares mappings with `decoder` removed.
- Frozen MLP BatchNorm buffers/parameters remain unchanged during backward; Task 1 checks buffers, `requires_grad`, and reconstruction gradients.
- Run records do not inflate iteration extrapolation; Task 3 provides a profile fixture containing A/B/run records.
- Decoder leading batch dimensions survive and output ends in ten; Task 1 uses `[2,3]` tissue and `[2,3,9]` timings.

---

### Task 1: Establish a common signal-decoder factory contract

**Files:**
- Create: `trad/modules/module_05_signal_decoder/decoder_factory.py`
- Modify: `trad/modules/module_07_objective_training/trad_trainer.py`
- Modify: `mlp/modules/module_07_objective_training/mlp_trainer.py`
- Test: `trad/tests/test_decoder_factory_contract.py`

**Interfaces:**
- Produces: `DecoderFactory = Callable[[QuantPointDataset, Mapping[str, Any], TradProtocol, torch.device], torch.nn.Module]`.
- Produces: `build_bloch_decoder(dataset, config, protocol, device) -> TradSignalSimulator`.
- Produces: `train_trad(..., decoder_factory: DecoderFactory | None = None, decoder_provenance: Mapping[str, Any] | None = None)`.
- Consumes: existing `load_frozen_mlp_decoder` in the MLP adapter.

- [ ] **Step 1: Write failing decoder-contract tests**

```python
def test_decoder_contract_preserves_leading_dimensions_and_ten_weights():
    t1 = torch.full((2, 3), 1000.0, requires_grad=True)
    t2 = torch.full((2, 3), 50.0, requires_grad=True)
    b1 = torch.ones((2, 3), requires_grad=True)
    timing = torch.full((2, 3, 9), 70.0)
    for decoder in (bloch_decoder, frozen_mlp_decoder):
        output = decoder(t1, t2, b1, timing, protocol, normalize=True)
        assert output.shape == (2, 3, 10)
        assert output.dtype == t1.dtype
        output.sum().backward(retain_graph=True)
    assert t1.grad is not None
```

- [ ] **Step 2: Verify RED**

Run: `conda run --no-capture-output -n knesvr_torch python -m pytest trad/tests/test_decoder_factory_contract.py -q`  
Expected: FAIL because the common factory contract does not exist.

- [ ] **Step 3: Implement the factory seam**

```python
DecoderFactory = Callable[[QuantPointDataset, Mapping[str, Any], TradProtocol, torch.device], nn.Module]

def build_bloch_decoder(dataset, config, protocol, device):
    return TradSignalSimulator().to(device)

def train_trad(..., decoder_factory: DecoderFactory | None = None, decoder_provenance=None):
    decoder = (build_bloch_decoder if decoder_factory is None else decoder_factory)(dataset, config, protocol, device)
```

Pass the decoder to `TradTrainingModel`; replace the MLP optimization loop with a thin adapter to this function. Keep FrozenMLP parameters out of optimizer groups.

- [ ] **Step 4: Verify GREEN**

Run: `conda run --no-capture-output -n knesvr_torch python -m pytest trad/tests/test_decoder_factory_contract.py mlp/tests/test_frozen_mlp_gradient.py mlp/tests/test_frozen_mlp_batchnorm_eval.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trad/modules/module_05_signal_decoder/decoder_factory.py trad/modules/module_07_objective_training/trad_trainer.py mlp/modules/module_07_objective_training/mlp_trainer.py trad/tests/test_decoder_factory_contract.py
git commit -m "Share reconstruction decoder factory"
```

### Task 2: Add controlled configuration and prove reconstruction parity

**Files:**
- Create: `mlp/configs/mlp_surrogate_v1/CYJ_B6.yaml`
- Modify: `mlp/scripts/reconstruct_subject.py`
- Modify: `mlp/scripts/run_training.py`
- Test: `mlp/tests/test_mlp_trad_orchestration_parity.py`
- Create: `MLP_SURROGATE_CHECKPOINT_INTEGRATION.md`

**Interfaces:**
- Consumes: `train_trad(..., decoder_factory=...)` from Task 1.
- Produces: `controlled_config_without_decoder(config: Mapping[str, Any]) -> dict[str, Any]`.
- Produces: MLP CLI requiring explicit `--mlp-checkpoint`.

- [ ] **Step 1: Write failing parity tests**

```python
def test_controlled_b6_configs_are_equal_except_decoder():
    assert controlled_config_without_decoder(resolve_config(TRAD_B6)) == controlled_config_without_decoder(resolve_config(MLP_B6))

def test_shared_route_has_equal_normalization_psf_loss_and_optimizer_groups(dataset):
    bloch = build_controlled_model(dataset, decoder_factory=build_bloch_decoder)
    mlp = build_controlled_model(dataset, decoder_factory=frozen_factory)
    assert bloch.intensity_scale.item() == mlp.intensity_scale.item()
    assert bloch.rigid_psf.group_resolution_xyz_mm.equal(mlp.rigid_psf.group_resolution_xyz_mm)
    assert optimizer_group_names(bloch) == optimizer_group_names(mlp)
```

- [ ] **Step 2: Verify RED**

Run: `conda run --no-capture-output -n knesvr_torch python -m pytest mlp/tests/test_mlp_trad_orchestration_parity.py -q`  
Expected: FAIL because the controlled configuration and shared-route adapter do not exist.

- [ ] **Step 3: Implement controlled route**

Create a B6-derived configuration with `decoder.kind: frozen_mlp` and `decoder.checkpoint: null`. Require CLI `--mlp-checkpoint`; pass it only to the strict loader. Reuse Trad normalization provenance, loss, and optimizer. Document the existing validation/manual approval sequence and the new output root without recommending metadata edits.

- [ ] **Step 4: Verify GREEN**

Run: `conda run --no-capture-output -n knesvr_torch python -m pytest mlp/tests/test_mlp_trad_orchestration_parity.py mlp/tests/test_mlp_checkpoint_compatibility.py mlp/tests/test_mlp_online_decoder_contract.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mlp/configs/mlp_surrogate_v1/CYJ_B6.yaml mlp/scripts/reconstruct_subject.py mlp/scripts/run_training.py mlp/tests/test_mlp_trad_orchestration_parity.py MLP_SURROGATE_CHECKPOINT_INTEGRATION.md
git commit -m "Configure controlled MLP surrogate route"
```

### Task 3: Extend the shared timing profile and auto-write summaries

**Files:**
- Modify: `trad/modules/module_07_objective_training/experiment_infrastructure.py`
- Modify: `trad/modules/module_07_objective_training/trad_trainer.py`
- Modify: `trad/scripts/run_training.py`
- Modify: `trad/evaluation/profiling/summarize_timing_profile.py`
- Test: `trad/tests/test_timing_profile_summary.py`
- Test: `trad/tests/test_experiment_infrastructure.py`

**Interfaces:**
- Produces: `RunProfiler.section(name)` and `RunProfiler.append_to(path)`.
- Produces: summary that accepts `stage="run"` records without adding them to A/B extrapolation.

- [ ] **Step 1: Write failing profile tests**

```python
def test_profile_contains_common_run_and_iteration_sections(tmp_path):
    sections = profile_sections(write_profile_with_run_samples(tmp_path))
    assert {"checkpoint_load", "data_loading", "initialization", "optimization_total", "validation", "export", "total_runtime"} <= sections
    assert {"forward_decoder", "loss", "backward", "optimizer_step"} <= sections

def test_summary_excludes_run_records_from_iteration_estimate(tmp_path):
    result = summarize_timing_profile(profile_with_run_records(tmp_path), stage_iterations={"A": 2, "B": 3})
    assert result["estimated_total_training_seconds"] == 0.05
```

- [ ] **Step 2: Verify RED**

Run: `conda run --no-capture-output -n knesvr_torch python -m pytest trad/tests/test_timing_profile_summary.py trad/tests/test_experiment_infrastructure.py -q`  
Expected: FAIL because run-level samples and automatic summaries are absent.

- [ ] **Step 3: Implement profile extension**

Keep CSV fields unchanged. Add the shared run collector; record `data_loading`, `initialization`, decoder `checkpoint_load`, `optimization_total`, `validation`, `export`, and `total_runtime`. Emit `forward_decoder` and `loss` aliases while preserving detailed legacy records. Call the existing summary writer after export for both adapters.

- [ ] **Step 4: Verify GREEN**

Run: `conda run --no-capture-output -n knesvr_torch python -m pytest trad/tests/test_timing_profile_summary.py trad/tests/test_experiment_infrastructure.py mlp/tests/test_mlp_timing_pool.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trad/modules/module_07_objective_training/experiment_infrastructure.py trad/modules/module_07_objective_training/trad_trainer.py trad/scripts/run_training.py trad/evaluation/profiling/summarize_timing_profile.py trad/tests/test_timing_profile_summary.py trad/tests/test_experiment_infrastructure.py
git commit -m "Unify reconstruction timing profiles"
```

### Task 4: Verify integration without a formal reconstruction

**Files:**
- Modify: `MLP_REPLACEMENT_INSPECTION.md`
- Modify: `MLP_SURROGATE_CHECKPOINT_INTEGRATION.md`
- Test: `mlp/tests/test_mlp_trad_orchestration_parity.py`

**Interfaces:**
- Consumes: Tasks 1–3 and the existing strict MLP checkpoint loader.
- Produces: an auditable no-run command template gated on `approved_by_manual_review`.

- [ ] **Step 1: Write failing gate-documentation test**

```python
def test_controlled_command_requires_explicit_approved_checkpoint():
    text = Path("MLP_SURROGATE_CHECKPOINT_INTEGRATION.md").read_text()
    assert "--mlp-checkpoint" in text
    assert "mlp_surrogate_v1/CYJ_B6" in text
    assert "approved_by_manual_review" in text
```

- [ ] **Step 2: Verify RED**

Run: `conda run --no-capture-output -n knesvr_torch python -m pytest mlp/tests/test_mlp_trad_orchestration_parity.py -q`  
Expected: FAIL until the gate document and controlled command exist.

- [ ] **Step 3: Document and verify**

State that no formal reconstruction ran; name the required approval status; include the command template with explicit output root. Run all Task 1–3 targeted tests and `git diff --check`; do not execute a training command.

- [ ] **Step 4: Verify GREEN**

Run: `conda run --no-capture-output -n knesvr_torch python -m pytest trad/tests/test_decoder_factory_contract.py mlp/tests/test_mlp_trad_orchestration_parity.py trad/tests/test_timing_profile_summary.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add MLP_REPLACEMENT_INSPECTION.md MLP_SURROGATE_CHECKPOINT_INTEGRATION.md trad/tests/test_decoder_factory_contract.py mlp/tests/test_mlp_trad_orchestration_parity.py
git commit -m "Document MLP surrogate experiment gate"
```
