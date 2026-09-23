import csv
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from modules.module_07_objective_training.experiment_infrastructure import (
    CachedBalancedSampler,
    FixedMonitorSet,
    IterationProfiler,
    normalize_step1_config,
    write_experiment_manifest,
)


def _dataset() -> SimpleNamespace:
    weights = torch.arange(10, dtype=torch.long).repeat_interleave(3)
    stacks = torch.arange(3, dtype=torch.long).repeat(10)
    size = weights.numel()
    return SimpleNamespace(
        xyz=torch.arange(size * 3, dtype=torch.float32).reshape(size, 3),
        v=torch.arange(size, dtype=torch.float32),
        group_idx=torch.zeros(size, dtype=torch.long),
        weight_idx=weights,
        stack_idx=stacks,
        timing=torch.zeros(size, 9),
        group_tr_ms=torch.tensor([2.61], dtype=torch.float64),
        group_vps=torch.tensor([87]),
    )


def test_cached_sampler_and_monitor_keep_all_weights_balanced_and_deterministic():
    dataset = _dataset()
    sampled = CachedBalancedSampler(dataset).sample(30)
    assert torch.equal(torch.bincount(sampled["weight_idx"], minlength=10), torch.full((10,), 3))
    first = FixedMonitorSet.from_dataset(dataset, seed=17, samples_per_weight_per_stack=1).batch
    second = FixedMonitorSet.from_dataset(dataset, seed=17, samples_per_weight_per_stack=1).batch
    assert torch.equal(first["xyz"], second["xyz"])
    assert torch.equal(torch.bincount(first["weight_idx"], minlength=10), torch.full((10,), 3))
    assert set(first["stack_idx"].tolist()) == {0, 1, 2}


def test_profiler_writes_raw_named_samples(tmp_path):
    profiler = IterationProfiler(torch.device("cpu"), every=1, warmup_samples=0)
    profiler.begin(1, "A")
    with profiler.section("batch_sampling"):
        time.sleep(0.001)
    profiler.end()
    profiler.write_csv(tmp_path / "timing_profile.csv")
    with (tmp_path / "timing_profile.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {"iteration", "stage", "section", "milliseconds"} == set(rows[0])
    assert {row["section"] for row in rows} == {"batch_sampling", "whole_iteration"}


def test_profiler_appends_common_run_level_sections_to_the_same_csv(tmp_path):
    profiler = IterationProfiler(torch.device("cpu"), every=0, warmup_samples=0)
    profile = tmp_path / "timing_profile.csv"
    profiler.write_csv(profile)
    profiler.append_run_sections(profile, {"checkpoint_load": 1.0, "data_loading": 2.0, "initialization": 3.0, "optimization_total": 4.0, "validation": 5.0, "export": 6.0, "total_runtime": 7.0})
    with profile.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["section"] for row in rows} == {"checkpoint_load", "data_loading", "initialization", "optimization_total", "validation", "export", "total_runtime"}
    assert {row["stage"] for row in rows} == {"run"}


def test_manifest_records_clean_and_dirty_git_provenance(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    for command in (("git", "init"), ("git", "config", "user.email", "test@example.com"), ("git", "config", "user.name", "Test")):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
    prepared = repo / "prepared.npz"; prepared.write_bytes(b"prepared")
    subprocess.run(("git", "add", "tracked.txt", "prepared.npz"), cwd=repo, check=True, capture_output=True)
    subprocess.run(("git", "commit", "-m", "initial"), cwd=repo, check=True, capture_output=True)
    output = tmp_path / "run"; output.mkdir()
    config = output / "config_resolved.yaml"; config.write_text("training: {}\n", encoding="utf-8")
    clean = write_experiment_manifest(output, route="trad_bloch", subject_id="CYJ", repo_root=repo, config_resolved=config, prepared_inputs=[prepared], protocol={"tr_ms": 2.61, "vps": 87}, stack_group_counts={"sax": 14}, seed=7, command="test")
    assert clean["git_dirty"] is False and clean["git_diff_sha256"] is None
    (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    dirty = write_experiment_manifest(output, route="trad_bloch", subject_id="CYJ", repo_root=repo, config_resolved=config, prepared_inputs=[prepared], protocol={"tr_ms": 2.61, "vps": 87}, stack_group_counts={"sax": 14}, seed=7, command="test")
    assert dirty["git_dirty"] is True and dirty["git_diff_sha256"]
    assert (output / "code_diff.patch").is_file()
    assert json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))["git_diff_sha256"] == dirty["git_diff_sha256"]


def test_step2_schema_exposes_disabled_by_default_optional_controls():
    config = normalize_step1_config({"training": {"psf_samples": 8, "learning_rates": {"variance": 0.001}}, "variance": {"enabled": True, "pixel": True, "slice": True}, "spatial_regularization": {"mode": "edge-preserving", "n_points": 7, "amplitude_guidance": {"enabled": True}}})
    assert config["variance"]["enabled"] is True
    assert config["spatial_regularization"]["mode"] == "edge-preserving"
    assert config["psf"]["export"]["enabled"] is False
