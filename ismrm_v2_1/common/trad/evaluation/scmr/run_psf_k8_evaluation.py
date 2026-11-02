"""Retired all-in-one PSF K=8 entry point.

Use ``run_psf_k8_gpu_stage`` on the CUDA/tiny-cuda-nn server, transfer the
result, then use ``run_psf_k8_postprocess`` locally for MATLAB and metrics.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parser.error(
        "PSF K=8 evaluation is split by design: run python -m "
        "trad.evaluation.scmr.run_psf_k8_gpu_stage on the CUDA server, then "
        "python -m trad.evaluation.scmr.run_psf_k8_postprocess locally."
    )


if __name__ == "__main__":
    main()
