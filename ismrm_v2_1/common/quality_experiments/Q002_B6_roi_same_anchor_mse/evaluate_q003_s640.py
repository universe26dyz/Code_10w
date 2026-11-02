"""Dedicated strict B6/Q002-S640/Q003-S640 evaluator."""

from __future__ import annotations

import argparse

from .evaluate_q002 import run_q003_s640_three_way


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--q002-s640-run-root", required=True)
    parser.add_argument("--b6-map-root", required=True)
    parser.add_argument("--b6-d2-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--preprocessed-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    print(run_q003_s640_three_way(build_parser().parse_args()))


if __name__ == "__main__":
    main()
