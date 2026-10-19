"""CPU contracts for the formal fixed-L100 matched-output pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean import matched_output


def _volume(path: Path, value: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.full((2, 1, 1), value, np.float32), np.eye(4)), path)


def test_matched_evaluator_rejects_central_vs_psf128() -> None:
    with pytest.raises(ValueError, match="endpoint mismatch"):
        matched_output.require_matched_endpoints("central", "psf128")


def test_matched_evaluator_rejects_psf128_vs_central() -> None:
    with pytest.raises(ValueError, match="endpoint mismatch"):
        matched_output.require_matched_endpoints("psf128", "central")


@pytest.mark.parametrize(("endpoint", "mode", "samples"), (("central", "k1", 1), ("psf128", "k128", 128)))
def test_fit_first_export_endpoint_contract_is_explicit(endpoint: str, mode: str, samples: int) -> None:
    spec = matched_output.endpoint_contract(endpoint)
    assert spec["fit_first_sampling_mode"] == mode
    assert spec["output_query_psf"]["n_inference_samples"] == samples
    assert spec["query_mask_semantics"] == "frozen_l100_checkpoint_mask_only"


def test_formal_manifest_distinguishes_training_export_and_nesvor_samples(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"; checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "sparse.json").write_text("{}", encoding="utf-8")
    l100_mask = tmp_path / "l100_mask.nii.gz"; _volume(l100_mask)
    manifest = matched_output.fixed_l100_export_manifest(
        subject="CYJ", condition="L25", endpoint="psf128", sparse_manifest=tmp_path / "sparse.json",
        r0_native_maps_root=tmp_path, sparse_checkpoint=checkpoint, l100_mask=l100_mask,
        training_route="Fit-first", training_psf={"nesvor_training_n_samples": 256},
    )
    assert manifest["training_psf"]["nesvor_training_n_samples"] == 256
    assert manifest["output_query_psf"]["n_inference_samples"] == 128
    assert manifest["endpoint"] == "psf128"
    assert manifest["query_mask_source"]["path"].endswith("l100_mask.nii.gz")
    assert manifest["no_training"] is True and manifest["no_refitting"] is True


def test_proposed_formalizer_uses_existing_central_and_posthoc_psf128_exports(tmp_path: Path) -> None:
    source = tmp_path / "proposed"; checkpoint = source / "model.pt"; checkpoint.parent.mkdir(); checkpoint.write_bytes(b"checkpoint")
    _volume(source / "T1_3D.nii.gz", 1000); _volume(source / "T2_3D.nii.gz", 50)
    _volume(source / "formal_export_psf128" / "T1_3D.nii.gz", 1001)
    _volume(source / "formal_export_psf128" / "T2_3D.nii.gz", 51)
    (source / "formal_export_psf128" / "formal_export_manifest.json").write_text(json.dumps({"n_inference_samples": 128}), encoding="utf-8")
    output = tmp_path / "formal"
    matched_output.formalize_proposed_export(subject="CYJ", condition="L25", source_root=source, endpoint="psf128", output=output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert (output / "T1_3D.nii.gz").is_file()
    assert manifest["export_kind"] == "post_hoc_map_domain_quantitative_export"
    assert manifest["output_query_psf"]["n_inference_samples"] == 128
    with pytest.raises(FileExistsError, match="overwrite"):
        matched_output.formalize_proposed_export(subject="CYJ", condition="L25", source_root=source, endpoint="psf128", output=output)


def test_boundary_discrepancy_keeps_l100_primary_but_marks_decomposition_inexact() -> None:
    exact = matched_output.primary_decomposition_contract(8, 0)
    inexact = matched_output.primary_decomposition_contract(8, 2)
    assert exact["primary_region"] == "l100_full" and exact["decomposition_exact"] is True
    assert inexact["primary_region"] == "l100_full" and inexact["decomposition_exact"] is False
    assert inexact["diagnostic_boundary_discrepancy_voxels"] == 2
