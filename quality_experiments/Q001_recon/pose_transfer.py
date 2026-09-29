"""Physical rigid-delta transfer for Q001B, never raw axis-angle subtraction."""

from __future__ import annotations

import torch

from trad.third_party.nesvor.nesvor.transform import RigidTransform


def compose_stack_delta(dicom_group_pose: torch.Tensor, initial_stack_pose: torch.Tensor, registered_stack_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``delta.compose(dicom_pose)`` in rotation-first, trans_first=True form."""

    dicom = RigidTransform(torch.as_tensor(dicom_group_pose, dtype=torch.float32), trans_first=True)
    initial = RigidTransform(torch.as_tensor(initial_stack_pose, dtype=torch.float32), trans_first=True)
    registered = RigidTransform(torch.as_tensor(registered_stack_pose, dtype=torch.float32), trans_first=True)
    delta = registered.compose(initial.inv())
    return delta.compose(dicom).axisangle(trans_first=True), delta.axisangle(trans_first=True)
