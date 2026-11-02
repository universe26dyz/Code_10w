"""Route-level separation between Q001A full reconstruction and Q001B transfer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class RoutePlan:
    experiment: str
    training_inputs: tuple[str, ...]
    stack_initialization_inputs: tuple[str, ...]
    stack_registration_calls: int


def stack_identity(path: str) -> str:
    """Return the native stack label encoded by a prepared observations path."""
    source = Path(path)
    label = source.parent.name if source.name == "observations.npz" else source.name
    if label not in {"sax", "2ch", "4ch"}:
        raise ValueError(f"Q001 prepared input has no supported stack identity: {path}")
    return label


def registration_inputs_for_training(training_inputs: Sequence[str], full_inputs: Sequence[str]) -> tuple[str, ...]:
    """Order full-FOV registration sources by training-stack identity, never list order."""
    full_by_stack = {stack_identity(path): path for path in full_inputs}
    if len(full_by_stack) != 3:
        raise ValueError("Q001 full registration inputs must contain each stack exactly once.")
    ordered = tuple(full_by_stack[stack_identity(path)] for path in training_inputs)
    if {stack_identity(path) for path in training_inputs} != set(full_by_stack):
        raise ValueError("Q001 cropped/full stack identities differ.")
    return ordered


def build_route_plan(mode: str, full_inputs: Sequence[str], cropped_inputs: Sequence[str]) -> RoutePlan:
    full, cropped = tuple(full_inputs), tuple(cropped_inputs)
    if len(full) != 3 or len(cropped) != 3: raise ValueError("Q001 routes require sax, 2ch, and 4ch observations.")
    if mode == "Q001A": return RoutePlan(mode, full, registration_inputs_for_training(full, full), 1)
    if mode == "Q001B": return RoutePlan(mode, cropped, registration_inputs_for_training(cropped, full), 1)
    raise ValueError(f"Unsupported Q001 route: {mode}")
