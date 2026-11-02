"""Inspect the explicit numeric v7.3 bridge contract without reading pixel DICOM data."""

from __future__ import annotations

import argparse

from .mat_v73 import load_preprocessed_v73


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-mat", required=True)
    args = parser.parse_args()
    data = load_preprocessed_v73(args.preprocessed_mat)
    print(f"Mag={data['mag'].shape}")
    print(f"Mag_crop={data['mag_crop'].shape}")
    print(f"Acq_time={data['acq_time_ms'].shape}")
    print(f"SOPInstanceUID_count={len(data['sop_instance_uids'])}")


if __name__ == "__main__":
    main()
