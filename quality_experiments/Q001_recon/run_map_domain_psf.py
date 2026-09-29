"""Route-aware post-hoc Q001 map-domain PSF K=32 reprojection; never reconstructs."""
from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from trad.evaluation.scmr.run_map_domain_psf_comparison import run as run_map_domain_psf


SEED = 20260911


def run(args: argparse.Namespace) -> Path:
    run_root, output = Path(args.run_root), Path(args.output)
    mode = args.mode
    if mode not in {"Q001A", "Q001B"}: raise ValueError("mode must be Q001A or Q001B")
    for path in (run_root / "T1_3D.nii.gz", run_root / "T2_3D.nii.gz", run_root / "final_rigid_poses.json"):
        if not path.is_file(): raise FileNotFoundError(f"Missing final Q001 export: {path}")
    return run_map_domain_psf(Namespace(method="shared_svr", subject_id="CYJ", t1_volume=str(run_root / "T1_3D.nii.gz"), t2_volume=str(run_root / "T2_3D.nii.gz"), n_samples=32, seed=SEED, output=str(output), native_reference=str(args.reference_root) if mode == "Q001B" else None, preprocessed_root=str(args.preprocessed_root) if mode == "Q001B" else None, mask_bundle=None, prepared_root=str(args.prepared_root), final_poses=str(run_root / "final_rigid_poses.json"), t1_registered_slices=None, t2_registered_slices=None, decoder_type="FrozenMLP", run_label=f"{mode}_true_map_domain_psf_K32"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("Q001A", "Q001B"), required=True); parser.add_argument("--run-root", required=True); parser.add_argument("--prepared-root", required=True); parser.add_argument("--reference-root", required=True); parser.add_argument("--preprocessed-root", required=True); parser.add_argument("--output", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__": main()
