"""Dedicated immutable Q004-S640 central-K1 MSE-only launcher."""

from __future__ import annotations

from .contracts import Q004S640_K1_EXPERIMENT_ID
from .run_q002_reconstruction import build_parser as _build_shared_parser
from .run_q002_reconstruction import run


def build_parser():
    parser = _build_shared_parser()
    parser.description = __doc__
    parser.set_defaults(experiment_id=Q004S640_K1_EXPERIMENT_ID, anchor_batch_size=640)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.experiment_id != Q004S640_K1_EXPERIMENT_ID or args.anchor_batch_size != 640:
        raise ValueError("Q004-S640-K1 requires its immutable experiment_id and anchor_batch_size=640.")
    print(run(args))


if __name__ == "__main__":
    main()
