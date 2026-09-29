"""Route-level separation between Q001A full reconstruction and Q001B transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RoutePlan:
    experiment: str
    training_inputs: tuple[str, ...]
    stack_initialization_inputs: tuple[str, ...]
    stack_registration_calls: int


def build_route_plan(mode: str, full_inputs: Sequence[str], cropped_inputs: Sequence[str]) -> RoutePlan:
    full, cropped = tuple(full_inputs), tuple(cropped_inputs)
    if len(full) != 3 or len(cropped) != 3: raise ValueError("Q001 routes require sax, 2ch, and 4ch observations.")
    if mode == "Q001A": return RoutePlan(mode, full, full, 1)
    if mode == "Q001B": return RoutePlan(mode, cropped, full, 1)
    raise ValueError(f"Unsupported Q001 route: {mode}")
