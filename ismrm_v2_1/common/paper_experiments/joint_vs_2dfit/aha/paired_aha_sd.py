"""Render AHA16 SD bullseyes from one completed paired final-pose CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .cyj11_postprocess import write_aha_sd_bullseyes


def build_paired_aha_sd(*, segment_statistics_csv: str | Path, output: str | Path) -> Path:
    """Publish reference, Proposed, and Fit-first SD bullseyes without recomputing AHA."""
    source = Path(segment_statistics_csv)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return write_aha_sd_bullseyes(rows, output, manifest_updates={
        "source_aha16_statistics": str(source.resolve()),
        "sampling": "final_pose",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-statistics-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build_paired_aha_sd(segment_statistics_csv=args.segment_statistics_csv, output=args.output))


if __name__ == "__main__":
    main()
