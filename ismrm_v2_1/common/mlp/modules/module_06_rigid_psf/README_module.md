# Module 06 — Rigid group pose and PSF forward

`GroupRigidPSF` owns exactly one vendored-NeSVoR axis-angle rigid pose per
native spatial `group_idx`.  It starts from the cropped HB1 affine expressed
as RAS-mm local-to-world geometry.  It exposes `deformable = false` and has no
deformation network or parameters.

PSF noise is anisotropic Gaussian noise in local `[column,row,slice]` mm axes,
with sigma from NeSVoR `resolution2sigma(..., isotropic=False)`, before the
group transform.  `TradQuantitativeForward` evaluates every PSF world sample,
simulates every sample's 10-HB fingerprint, selects the observed weight,
multiplies by `A`, and only then averages samples.

`axisangle_init` 是不可训练 buffer，保存 DICOM-derived initial pose。Stage B
rigid regularization 复用 NeSVoR `trans_loss`：相对 init 的 rotation MSE 加
`1e-3*spatial_scaling^2` 倍 translation MSE。训练时该模块接收
centered/scaled local axes、resolution 与 pose；physical pose 的唯一可逆转换在
Module 07 training-space adapter 中完成。
