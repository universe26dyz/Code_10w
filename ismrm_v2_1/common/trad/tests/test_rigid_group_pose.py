import torch

from modules.module_06_rigid_psf.rigid_psf_forward import GroupRigidPSF


def test_rigid_pose_is_one_axisangle_tensor_per_group_and_has_no_deformation():
    rigid = GroupRigidPSF(torch.zeros((2, 6)), torch.tensor([[1.0, 2.0, 6.0], [1.5, 1.5, 8.0]]))
    assert rigid.group_count == 2
    assert rigid.axisangle.shape == (2, 6)
    assert rigid.deformable is False
    assert all("deform" not in name.lower() for name, _ in rigid.named_parameters())
    transformed = rigid.transform_local_to_ras(torch.zeros((3, 3)), torch.tensor([0, 1, 1], dtype=torch.long))
    assert torch.isfinite(transformed).all()
