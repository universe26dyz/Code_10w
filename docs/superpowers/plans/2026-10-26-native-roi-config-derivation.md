# Native ROI Config Derivation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive immutable, exact native-SAX ROI configs for every subject that has the canonical four-file manual ROI source.

**Architecture:** Extend the current ROI adapter with a config-only derivation function.  It reuses native-cache loading, composite/manual geometry admission, and exact lattice crop helpers; existing `derive-roi` subsequently validates and materializes the bundle without format changes.

**Tech Stack:** Python, NumPy, NiBabel, existing `evaluation_common` ROI helpers, pytest.

**Spec:** `docs/superpowers/specs/2026-10-26-native-roi-config-derivation-design.md`

## Global Constraints

- No interpolation, reconstruction, formal metric, Final5, or legacy-pipeline changes.
- Require one unambiguous, in-bounds integer signed-permutation mapping for every native SAX group.
- Write only a new immutable config; retain existing `derive-roi` as the bundle-writing verifier.

## Review Focus

- A native plane whose continuous transform is near, but not on, an integer lattice must fail; test it through the existing exact crop helper path.
- A source ROI with manual triad geometry disagreement must fail before config output; test no output exists.
- A missing or duplicated manual ROI slice mapping must fail; test an ambiguous derived mapping is rejected.
- A prepared SAX hash mismatch must remain rejected by native-cache loading; retain existing loader coverage.
- A successful config must contain all group IDs in native order and valid SHA256 provenance; assert the written schema fields.

---

### Task 1: Exact config derivation and CLI

**Files:**
- Modify: `paper_experiments/joint_vs_2dfit/aha/cyj11_legacy_roi.py`
- Modify: `paper_experiments/joint_vs_2dfit/aha/run_cyj11_aha_roi.py`
- Modify: `paper_experiments/joint_vs_2dfit/tests/test_cyj11_aha_roi.py`

**Interfaces:**
- Consumes: unified analysis root, subject ID, canonical source ROI directory, output config path.
- Produces: `derive_cyj11_native_roi_config(*, analysis_root: str | Path, subject: str, source_roi: str | Path, output: str | Path) -> Path` and CLI mode `derive-roi-config`.

- [ ] **Step 1: Write the failing test**

```python
def test_cyj11_config_derivation_writes_all_exact_native_plane_mappings(tmp_path, monkeypatch):
    config = derive_cyj11_native_roi_config(
        analysis_root=root, subject="DYZ", source_roi=source, output=output,
    )
    payload = json.loads(config.read_text())
    assert [plane["native_group_idx"] for plane in payload["planes"]] == [0, 1, 2]
    assert [plane["roi_slice_index"] for plane in payload["planes"]] == [2, 1, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n knesvr_torch python -m pytest paper_experiments/joint_vs_2dfit/tests/test_cyj11_aha_roi.py -q`

Expected: FAIL because the config-derivation interface does not exist.

- [ ] **Step 3: Implement exact config derivation and CLI dispatch**

Implement the stated function in `cyj11_legacy_roi.py`.  Load the cached native
SAX data through `load_cyj11_native_sax`, validate the manual triad plus
composite geometry with existing helpers, derive each slice via
`_native_plane_indices`, and write schema `cyj11_native_roi_derivation_config/v1`
with `prepared_sax`, `source_roi.path`, `native_sax_root`, and all group/slice
entries.  Add `derive-roi-config` and its required arguments to the CLI.

- [ ] **Step 4: Run targeted verification**

Run: `conda run --no-capture-output -n knesvr_torch python -m pytest paper_experiments/joint_vs_2dfit/tests/test_cyj11_aha_roi.py -q`

Expected: PASS.

- [ ] **Step 5: Run the repository test command and record existing unrelated collection failures**

Run: `conda run --no-capture-output -n knesvr_torch python -m pytest -q`

Expected: CYJ targeted suite remains green; repository collection may retain the
known unrelated `mlp`/`trad` module-collision errors.

- [ ] **Step 6: Commit**

```bash
git add paper_experiments/joint_vs_2dfit/aha/cyj11_legacy_roi.py \
  paper_experiments/joint_vs_2dfit/aha/run_cyj11_aha_roi.py \
  paper_experiments/joint_vs_2dfit/tests/test_cyj11_aha_roi.py \
  docs/superpowers/specs/2026-10-26-native-roi-config-derivation-design.md \
  docs/superpowers/plans/2026-10-26-native-roi-config-derivation.md
git commit -m "feat: derive subject-native ROI configs"
```
