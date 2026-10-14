"""Retired Phase-1.1 entry point retained only as an explicit safety tombstone."""

from __future__ import annotations


def retired_phase1_1(*_args, **_kwargs) -> None:
    raise RuntimeError(
        "Phase-1.1 composite GT is invalid for scientific use. "
        "Use generate_phase1_2_xcat3d.py with a validated single-source XCAT3D manifest."
    )


def main() -> None:
    retired_phase1_1()


if __name__ == "__main__":
    main()
