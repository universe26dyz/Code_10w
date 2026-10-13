"""Synthetic mesh-GT and real native-reference agreement metrics."""
from __future__ import annotations
import numpy as np

def _world_sample(volume: np.ndarray, volume_affine: np.ndarray, world: np.ndarray, order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import map_coordinates
    vox = np.linalg.inv(np.asarray(volume_affine, float)) @ world.reshape(4, -1)
    inside = np.all((vox[:3] >= 0) & (vox[:3] <= (np.asarray(volume.shape)[:, None] - 1)), axis=0)
    values = map_coordinates(np.asarray(volume, float), vox[:3], order=order, mode='constant', cval=np.nan).reshape(world.shape[1:])
    return values, inside.reshape(world.shape[1:])

def resample_volume_to_grid(volume: np.ndarray, volume_affine: np.ndarray, grid_shape: tuple[int, int, int], grid_affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear world-coordinate resampling onto the frozen 1-mm mesh grid."""
    indices = np.indices(grid_shape, dtype=float).reshape(3, -1)
    world = np.asarray(grid_affine, float) @ np.vstack((indices, np.ones(indices.shape[1])))
    return _world_sample(volume, volume_affine, world.reshape(4, *grid_shape), order=1)

def reproject_volume_to_native_plane(volume: np.ndarray, volume_affine: np.ndarray, native_affine: np.ndarray, plane_shape: tuple[int, int], slice_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Linear world-coordinate reprojection to one current native plane."""
    row, col = np.indices(plane_shape, dtype=float)
    indices = np.vstack((row.ravel(), col.ravel(), np.full(row.size, slice_index), np.ones(row.size)))
    world = np.asarray(native_affine, float) @ indices
    values, valid = _world_sample(volume, volume_affine, world.reshape(4, *plane_shape), order=1)
    return values, valid

def _metrics(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray, constant_normalization: float | None = None) -> dict:
    r,p=np.asarray(reference,float)[mask],np.asarray(prediction,float)[mask]
    if not len(r): raise ValueError("empty evaluation support")
    diff=p-r; rmse=float(np.sqrt(np.mean(diff**2))); rng=float(np.max(r)-np.min(r))
    pearson=float(np.corrcoef(r,p)[0,1]) if len(r)>1 and np.std(r)>0 and np.std(p)>0 else float("nan")
    ncc=float(np.dot(r-r.mean(),p-p.mean())/(np.linalg.norm(r-r.mean())*np.linalg.norm(p-p.mean()))) if np.std(r)>0 and np.std(p)>0 else float("nan")
    constant = np.std(r) == 0
    return {"bias":float(diff.mean()),"mae":float(np.abs(diff).mean()),"rmse":rmse,"relative_nrmse":rmse/(float(constant_normalization) if constant_normalization is not None else (rng if rng else float('nan'))),"valid_mape":float(np.mean(np.abs(diff[r!=0]/r[r!=0]))*100) if np.any(r!=0) else float("nan"),"pearson_r":None if constant else pearson,"ncc":None if constant else ncc,"correlation_status":"not_applicable_constant_reference" if constant else "applicable","ssim":None,"support_voxels":int(len(r))}

def synthetic_metrics(reference: np.ndarray, prediction: np.ndarray, myocardium_full: np.ndarray, valid: np.ndarray, normalization: float | None = None) -> dict:
    support=np.asarray(myocardium_full,bool)&np.asarray(valid,bool)&np.isfinite(reference)&np.isfinite(prediction)
    report=_metrics(reference,prediction,support,normalization); report.update({"primary_region":"full_mesh_myocardium","coverage_fraction":float(support.sum()/np.asarray(myocardium_full,bool).sum()),"out_of_grid_voxels":int((np.asarray(myocardium_full,bool)&~np.asarray(valid,bool)).sum()),"world_alignment":"raw known world geometry primary; common SE(3) diagnostic only"}); return report

def real_native_agreement(reference: np.ndarray, prediction: np.ndarray, myocardium_mask: np.ndarray) -> dict:
    report=_metrics(reference,prediction,np.asarray(myocardium_mask,bool)&np.isfinite(reference)&np.isfinite(prediction)); report["endpoint"]="native-reference agreement/consistency, not accuracy versus ground truth"; return report

def main() -> None:
    import argparse, json, nibabel as nib
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--mode', choices=('synthetic','real'), required=True); parser.add_argument('--synthetic-t1'); parser.add_argument('--synthetic-t2'); parser.add_argument('--gt-maps'); parser.add_argument('--mesh-myo'); parser.add_argument('--subject'); parser.add_argument('--real-t1'); parser.add_argument('--real-t2'); parser.add_argument('--native-reference'); parser.add_argument('--myocardium-bundle'); parser.add_argument('--output', required=True); args=parser.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError(f'refusing overwrite: {output}')
    if args.mode == 'real':
        if not all((args.subject,args.real_t1,args.real_t2,args.native_reference,args.myocardium_bundle)): raise ValueError('real evaluation requires subject, T1/T2 volumes, current reference, and masks')
        ref_root=Path(args.native_reference); reference=np.load(ref_root/'sax.npz'); masks=np.load(Path(args.myocardium_bundle)/'sax_myocardium_masks.npz'); result={}
        for parameter,volume_path in (('t1_ms',args.real_t1),('t2_ms',args.real_t2)):
            image=nib.load(volume_path); native=nib.load(str(ref_root/('sax_T1_stack_ms.nii.gz' if parameter=='t1_ms' else 'sax_T2_stack_ms.nii.gz'))); rows=[]
            for group in range(reference[parameter].shape[0]):
                predicted, valid = reproject_volume_to_native_plane(image.get_fdata(),image.affine,native.affine,reference[parameter].shape[1:],group)
                rows.append(real_native_agreement(reference[parameter][group],predicted,masks['myocardium_full'][group].astype(bool)&valid))
            result[parameter]={'per_native_sax_plane':rows}
        output.parent.mkdir(parents=True,exist_ok=False); output.write_text(json.dumps({'subject':args.subject,'endpoint':'agreement/consistency, not 3-D ground-truth accuracy','metrics':result},indent=2)); print(output); return
    if not all((args.synthetic_t1,args.synthetic_t2,args.gt_maps,args.mesh_myo)): raise ValueError('synthetic evaluation requires T1/T2 NIfTI, gt maps, and mesh mask')
    gt=np.load(args.gt_maps); mesh=np.load(args.mesh_myo); affine=np.eye(4); affine[:3,:3]=np.diag(mesh['spacing_mm']); affine[:3,3]=mesh['origin_mm']
    result={}
    for parameter,path,normal in (('t1_ms',args.synthetic_t1,1410.0),('t2_ms',args.synthetic_t2,50.0)):
        image=nib.load(path); prediction,valid=resample_volume_to_grid(image.get_fdata(),image.affine,gt[parameter].shape,affine); result[parameter]=synthetic_metrics(gt[parameter],prediction,mesh['myocardium_mask'],valid,normal)
    output.parent.mkdir(parents=True,exist_ok=False); output.write_text(json.dumps({'endpoint':'synthetic mesh myocardium; primary NRMSE uses fixed GT value','metrics':result},indent=2)); print(output)

if __name__ == '__main__': main()
