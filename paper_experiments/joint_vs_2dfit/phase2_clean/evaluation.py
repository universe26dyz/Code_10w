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

def _metrics(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict:
    r,p=np.asarray(reference,float)[mask],np.asarray(prediction,float)[mask]
    if not len(r): raise ValueError("empty evaluation support")
    diff=p-r; rmse=float(np.sqrt(np.mean(diff**2))); rng=float(np.max(r)-np.min(r))
    pearson=float(np.corrcoef(r,p)[0,1]) if len(r)>1 and np.std(r)>0 and np.std(p)>0 else float("nan")
    ncc=float(np.dot(r-r.mean(),p-p.mean())/(np.linalg.norm(r-r.mean())*np.linalg.norm(p-p.mean()))) if np.std(r)>0 and np.std(p)>0 else float("nan")
    return {"bias":float(diff.mean()),"mae":float(np.abs(diff).mean()),"rmse":rmse,"nrmse":rmse/rng if rng else float("nan"),"valid_mape":float(np.mean(np.abs(diff[r!=0]/r[r!=0]))*100) if np.any(r!=0) else float("nan"),"pearson_r":pearson,"ncc":ncc,"ssim":None,"support_voxels":int(len(r))}

def synthetic_metrics(reference: np.ndarray, prediction: np.ndarray, myocardium_full: np.ndarray, valid: np.ndarray) -> dict:
    support=np.asarray(myocardium_full,bool)&np.asarray(valid,bool)&np.isfinite(reference)&np.isfinite(prediction)
    report=_metrics(reference,prediction,support); report.update({"primary_region":"full_mesh_myocardium","coverage_fraction":float(support.sum()/np.asarray(myocardium_full,bool).sum()),"out_of_grid_voxels":int((np.asarray(myocardium_full,bool)&~np.asarray(valid,bool)).sum()),"world_alignment":"raw known world geometry primary; common SE(3) diagnostic only"}); return report

def real_native_agreement(reference: np.ndarray, prediction: np.ndarray, myocardium_mask: np.ndarray) -> dict:
    report=_metrics(reference,prediction,np.asarray(myocardium_mask,bool)&np.isfinite(reference)&np.isfinite(prediction)); report["endpoint"]="native-reference agreement/consistency, not accuracy versus ground truth"; return report

def main() -> None:
    import argparse, json
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--contract', required=True); args=parser.parse_args()
    print(json.dumps({'evaluation_contract': Path(args.contract).resolve().as_posix(), 'status':'inputs_required_for_evaluation'}, indent=2))

if __name__ == '__main__': main()
