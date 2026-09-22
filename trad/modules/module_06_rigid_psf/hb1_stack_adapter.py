"""Lossless prepared-HB1 view for vendored NeSVoR stack registration only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from trad.modules.module_02_data_bridge.geometry import apply_affine_rc, cropped_affine_lps_rc_to_initial_rigid, lps_to_ras
from trad.third_party.nesvor.nesvor.image import Stack
from trad.third_party.nesvor.nesvor.svr.registration import stack_registration
from trad.third_party.nesvor.nesvor.transform import RigidTransform, ax_transform_points


@dataclass(frozen=True)
class HB1GeometryTolerance:
    """DICOM geometry tolerances; values exceed observed float32-affine noise only."""

    world_mm: float = 2e-4
    orientation_deg: float = 0.05
    gap_mm: float = 1e-4


HB1_GEOMETRY_TOLERANCE = HB1GeometryTolerance()


@dataclass(frozen=True)
class HB1StackAudit:
    path: Path
    selected_observation_idx: np.ndarray
    sorted_group_ids: np.ndarray
    sorted_observation_idx: np.ndarray
    slice_positions_mm: np.ndarray
    gap_values_mm: np.ndarray
    gap_mm: float
    regular_gap: bool
    shape: tuple[int, int]
    spacing_rc_mm: tuple[float, float]
    thickness_mm: float
    max_row_angle_deg: float
    max_col_angle_deg: float
    input_order_monotonic: bool


@dataclass(frozen=True)
class HB1RegistrationStack:
    stack: Stack
    source_path: Path
    group_ids: np.ndarray
    observation_idx: np.ndarray
    images: torch.Tensor
    masks: torch.Tensor
    affines_lps_rc: np.ndarray
    spacing_rc_mm: np.ndarray
    thickness_mm: np.ndarray


@dataclass(frozen=True)
class RoundTripResult:
    max_error_mm: float
    mean_error_mm: float


@dataclass(frozen=True)
class StackInitialization:
    """Registration-only stack initialization, indexed by global native group."""

    dicom_axisangle_physical: torch.Tensor
    post_stack_init_axisangle_physical: torch.Tensor
    stack_pose_records: tuple[dict[str, object], ...]


def _load(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        required = {"images", "masks", "group_idx", "weight_idx", "affine_lps_rc", "pixel_spacing_rc_mm", "slice_thickness_mm"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Prepared observations lacks required HB1 fields: {sorted(missing)}")
        return {key: np.asarray(data[key]) for key in required}


def audit_hb1_stack(path: str | Path, *, tolerance: HB1GeometryTolerance = HB1_GEOMETRY_TOLERANCE) -> HB1StackAudit:
    """Select one dense weight-0 image per group and prove NeSVoR Stack geometry."""

    source, data = Path(path), _load(path)
    images, masks = np.asarray(data["images"]), np.asarray(data["masks"], dtype=bool)
    groups, weights = np.asarray(data["group_idx"], dtype=np.int64), np.asarray(data["weight_idx"], dtype=np.int64)
    affine, spacing, thickness = np.asarray(data["affine_lps_rc"], dtype=np.float64), np.asarray(data["pixel_spacing_rc_mm"], dtype=np.float64), np.asarray(data["slice_thickness_mm"], dtype=np.float64)
    group_ids = np.unique(groups)
    selected = []
    for group in group_ids:
        choices = np.flatnonzero((groups == group) & (weights == 0))
        if choices.size != 1:
            raise ValueError(f"{source}: group {int(group)} has {choices.size} HB1/weight-0 images; expected exactly one.")
        selected.append(int(choices[0]))
    selected_idx = np.asarray(selected, dtype=np.int64)
    hws = {tuple(images[index].shape) for index in selected_idx}
    if len(hws) != 1 or images.ndim != 3 or masks.shape != images.shape:
        raise ValueError(f"{source}: HB1 images/masks must share one dense [H,W] shape, found {sorted(hws)}.")
    spacing_selected, thickness_selected = spacing[selected_idx], thickness[selected_idx]
    if not np.allclose(spacing_selected, spacing_selected[0], rtol=0, atol=tolerance.world_mm):
        raise ValueError(f"{source}: HB1 in-plane pixel spacing is not compatible within {tolerance.world_mm} mm.")
    if not np.allclose(thickness_selected, thickness_selected[0], rtol=0, atol=tolerance.world_mm):
        raise ValueError(f"{source}: HB1 slice thickness is not compatible within {tolerance.world_mm} mm.")
    row = affine[selected_idx, :3, 0] / spacing_selected[:, :1]
    col = affine[selected_idx, :3, 1] / spacing_selected[:, 1:2]
    normal = np.cross(col[0], row[0]); normal /= np.linalg.norm(normal)
    row_angle = np.degrees(np.arccos(np.clip(row @ row[0], -1.0, 1.0)))
    col_angle = np.degrees(np.arccos(np.clip(col @ col[0], -1.0, 1.0)))
    if np.max(row_angle) > tolerance.orientation_deg or np.max(col_angle) > tolerance.orientation_deg:
        raise ValueError(f"{source}: HB1 orientations exceed {tolerance.orientation_deg} degrees (row={np.max(row_angle)}, col={np.max(col_angle)}).")
    h, w = next(iter(hws))
    centres = affine[selected_idx, :3, 3] + ((h - 1) / 2) * affine[selected_idx, :3, 0] + ((w - 1) / 2) * affine[selected_idx, :3, 1]
    positions = centres @ normal
    order = np.argsort(positions)
    sorted_idx, sorted_groups, sorted_pos = selected_idx[order], group_ids[order], positions[order]
    gaps = np.diff(sorted_pos)
    if gaps.size and np.any(gaps <= tolerance.gap_mm):
        raise ValueError(f"{source}: HB1 has duplicate/non-increasing slice locations (min gap={float(gaps.min())} mm).")
    gap = float(np.median(gaps)) if gaps.size else float(thickness_selected[0])
    regular = bool(not gaps.size or np.max(np.abs(gaps - gap)) <= tolerance.gap_mm)
    if not regular:
        raise ValueError(f"{source}: HB1 slice gaps cannot use one Stack.gap={gap} mm within {tolerance.gap_mm} mm (range {float(gaps.min())}..{float(gaps.max())}).")
    return HB1StackAudit(source, selected_idx, sorted_groups, sorted_idx, sorted_pos, gaps, gap, regular, (h, w), tuple(float(value) for value in spacing_selected[0]), float(thickness_selected[0]), float(np.max(row_angle)), float(np.max(col_angle)), bool(np.all(np.diff(positions) > 0)))


def build_hb1_registration_stack(path: str | Path, audit: HB1StackAudit, *, device: str | torch.device = "cpu") -> HB1RegistrationStack:
    """Create a lossless NeSVoR registration view; source observations remain untouched."""

    if Path(path) != audit.path:
        raise ValueError("Audit path does not match adapter source path.")
    data = _load(path)
    idx = audit.sorted_observation_idx
    images = torch.as_tensor(data["images"][idx], dtype=torch.float32, device=device)
    masks = torch.as_tensor(data["masks"][idx], dtype=torch.bool, device=device)
    affine, spacing, thickness = data["affine_lps_rc"][idx], data["pixel_spacing_rc_mm"][idx], data["slice_thickness_mm"][idx]
    transforms = []
    for current_affine, current_spacing, current_thickness in zip(affine, spacing, thickness):
        resolution_xyz = np.asarray([current_spacing[1], current_spacing[0], current_thickness], dtype=np.float64)
        transforms.append(cropped_affine_lps_rc_to_initial_rigid(current_affine, audit.shape, resolution_xyz, device=device).matrix(trans_first=True))
    transform = RigidTransform(torch.cat(transforms), trans_first=True)
    stack = Stack(images[:, None], masks[:, None], transform, resolution_x=float(audit.spacing_rc_mm[1]), resolution_y=float(audit.spacing_rc_mm[0]), thickness=audit.thickness_mm, gap=audit.gap_mm, name=audit.path.parent.name)
    return HB1RegistrationStack(stack, audit.path, audit.sorted_group_ids.copy(), idx.copy(), images, masks, np.asarray(affine, dtype=np.float64), np.asarray(spacing, dtype=np.float64), np.asarray(thickness, dtype=np.float64))


def round_trip_stack_geometry(adapted: HB1RegistrationStack) -> RoundTripResult:
    """Compare prepared affine RAS coordinates against every adapter slice at fixed pixels."""

    h, w = adapted.images.shape[-2:]
    rows = np.asarray([0.0, 0.0, h - 1.0, h - 1.0, (h - 1) / 2, 1.0, h - 2.0])
    cols = np.asarray([0.0, w - 1.0, 0.0, w - 1.0, (w - 1) / 2, w - 2.0, 1.0])
    errors = []
    for index, (affine, spacing) in enumerate(zip(adapted.affines_lps_rc, adapted.spacing_rc_mm)):
        local = torch.as_tensor(np.column_stack(((cols - (w - 1) / 2) * spacing[1], (rows - (h - 1) / 2) * spacing[0], np.zeros(rows.size))), dtype=torch.float32, device=adapted.stack.device)
        actual = ax_transform_points(adapted.stack.transformation.axisangle(trans_first=True)[index : index + 1].expand(rows.size, -1), local, trans_first=True).detach().cpu().numpy()
        expected = lps_to_ras(apply_affine_rc(affine, rows, cols))
        errors.extend(np.linalg.norm(actual - expected, axis=1).tolist())
    return RoundTripResult(float(np.max(errors)), float(np.mean(errors)))


def initialize_group_poses_from_hb1(
    dicom_axisangle_physical: torch.Tensor,
    prepared_inputs: list[str | Path],
    *,
    device: str | torch.device = "cpu",
    args_registration: dict[str, object] | None = None,
) -> StackInitialization:
    """Apply vendored stack registration as one rigid delta per prepared stack.

    The adapter view contains only HB1 images/masks.  This function neither
    rewrites a prepared NPZ nor alters the quantitative 10-weight dataset; it
    returns replacement *initial* poses for subsequent Stage A/Stage B use.
    """

    dicom = torch.as_tensor(dicom_axisangle_physical, dtype=torch.float32, device=device).detach().clone()
    if dicom.ndim != 2 or dicom.shape[1] != 6:
        raise ValueError("dicom_axisangle_physical must be [G,6].")
    if not prepared_inputs:
        raise ValueError("Stack initialization requires one or more prepared observations paths.")
    audits = [audit_hb1_stack(path) for path in prepared_inputs]
    adapters = [build_hb1_registration_stack(path, audit, device=device) for path, audit in zip(prepared_inputs, audits)]
    expected_groups = sum(len(adapter.group_ids) for adapter in adapters)
    if expected_groups != dicom.shape[0]:
        raise ValueError(f"Prepared HB1 groups ({expected_groups}) do not match quantitative pose count ({dicom.shape[0]}).")
    before = [adapter.stack.transformation.clone() for adapter in adapters]
    images = [adapter.stack.slices.clone() for adapter in adapters]
    masks = [adapter.stack.mask.clone() for adapter in adapters]
    registered = stack_registration([[adapter.stack for adapter in adapters]], args_registration=args_registration)
    post = dicom.clone()
    records: list[dict[str, object]] = []
    offset = 0
    for adapter, initial_transform, initial_images, initial_masks, registered_stack in zip(adapters, before, images, masks, registered):
        if not torch.equal(adapter.stack.slices, initial_images) or not torch.equal(adapter.stack.mask, initial_masks):
            raise RuntimeError("Vendored stack registration changed HB1 pixel or mask values.")
        local_groups = adapter.group_ids.astype(np.int64)
        expected_local = np.arange(local_groups.size, dtype=np.int64)
        if not np.array_equal(np.sort(local_groups), expected_local):
            raise ValueError(f"{adapter.source_path}: source group ids must be contiguous zero-based.")
        # The stack-level rigid delta is deliberately composed onto every
        # native group, preserving the shared pose of its ten weights.
        delta = registered_stack.transformation.mean().compose(initial_transform.mean().inv())
        group_index = torch.as_tensor(offset + local_groups, dtype=torch.long, device=dicom.device)
        post[group_index] = delta.compose(RigidTransform(dicom[group_index], trans_first=True)).axisangle(trans_first=True)
        initial_pose = initial_transform.mean().axisangle(trans_first=True)[0]
        final_pose = registered_stack.transformation.mean().axisangle(trans_first=True)[0]
        delta_axisangle = delta.axisangle(trans_first=True)[0]
        records.append({
            "prepared_input": str(adapter.source_path),
            "group_offset": int(offset),
            "group_count": int(local_groups.size),
            "dicom_initial_stack_pose_axisangle": initial_pose.detach().cpu().tolist(),
            "post_stack_init_pose_axisangle": final_pose.detach().cpu().tolist(),
            "translation_delta_mm": float(torch.linalg.vector_norm(delta_axisangle[3:]).detach().cpu()),
            "rotation_delta_deg": float(torch.linalg.vector_norm(delta_axisangle[:3]).detach().cpu() * 180.0 / np.pi),
        })
        offset += local_groups.size
    return StackInitialization(dicom.detach().cpu(), post.detach().cpu(), tuple(records))
