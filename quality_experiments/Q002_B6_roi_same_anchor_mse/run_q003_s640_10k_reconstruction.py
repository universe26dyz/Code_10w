"""Dedicated immutable Q003-S640 6k primary plus 10k continuation launcher."""

from __future__ import annotations

from .contracts import Q003S640_10K_EXPERIMENT_ID
from .run_q002_reconstruction import build_parser as _build_shared_parser
from .run_q002_reconstruction import run


def build_parser():
    parser = _build_shared_parser()
    parser.description = __doc__
    parser.set_defaults(experiment_id=Q003S640_10K_EXPERIMENT_ID, anchor_batch_size=640)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.experiment_id != Q003S640_10K_EXPERIMENT_ID or args.anchor_batch_size != 640:
        raise ValueError("Q003-S640-10k requires its immutable experiment_id and anchor_batch_size=640.")
    print(run(args))


if __name__ == "__main__":
    main()
