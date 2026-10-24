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


def reproject_binary_mask_to_native_plane(mask: np.ndarray, mask_affine: np.ndarray, native_affine: np.ndarray, plane_shape: tuple[int, int], slice_index: int = 0) -> np.ndarray:
    """Map a formal query mask to native geometry with nearest-neighbour sampling."""
    row, col = np.indices(plane_shape, dtype=float)
    indices = np.vstack((row.ravel(), col.ravel(), np.full(row.size, slice_index), np.ones(row.size)))
    world = np.asarray(native_affine, float) @ indices
    values, inside = _world_sample(mask, mask_affine, world.reshape(4, *plane_shape), order=0)
    return np.asarray(inside, bool) & np.isfinite(values) & (np.asarray(values) > .5)

def _metrics(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray, constant_normalization: float | None = None) -> dict:
    r,p=np.asarray(reference,float)[mask],np.asarray(prediction,float)[mask]
    if not len(r): raise ValueError("empty evaluation support")
    diff=p-r; rmse=float(np.sqrt(np.mean(diff**2))); rng=float(np.max(r)-np.min(r))
    pearson=float(np.corrcoef(r,p)[0,1]) if len(r)>1 and np.std(r)>0 and np.std(p)>0 else float("nan")
    ncc=float(np.dot(r-r.mean(),p-p.mean())/(np.linalg.norm(r-r.mean())*np.linalg.norm(p-p.mean()))) if np.std(r)>0 and np.std(p)>0 else float("nan")
    constant = np.std(r) == 0
    return {"bias":float(diff.mean()),"mae":float(np.abs(diff).mean()),"rmse":rmse,"relative_nrmse":rmse/(float(constant_normalization) if constant_normalization is not None else (rng if rng else float('nan'))),"valid_mape":float(np.mean(np.abs(diff[r!=0]/r[r!=0]))*100) if np.any(r!=0) else float("nan"),"pearson_r":None if constant else pearson,"ncc":None if constant else ncc,"correlation_status":"not_applicable_constant_reference" if constant else "applicable","ssim":None,"ssim_status":"not_implemented","support_voxels":int(len(r)),"sum_squared_error":float(np.sum(diff**2)),"sum_absolute_error":float(np.sum(np.abs(diff))),"sum_signed_error":float(np.sum(diff)),"reference_mean_ms":float(r.mean()),"prediction_mean_ms":float(p.mean()),"mean_difference_ms":float(diff.mean())}

def synthetic_metrics(reference: np.ndarray, prediction: np.ndarray, myocardium_full: np.ndarray, valid: np.ndarray, normalization: float | None = None) -> dict:
    support=np.asarray(myocardium_full,bool)&np.asarray(valid,bool)&np.isfinite(reference)&np.isfinite(prediction)
    report=_metrics(reference,prediction,support,normalization); report.update({"primary_region":"full_mesh_myocardium","coverage_fraction":float(support.sum()/np.asarray(myocardium_full,bool).sum()),"out_of_grid_voxels":int((np.asarray(myocardium_full,bool)&~np.asarray(valid,bool)).sum()),"world_alignment":"raw known world geometry primary; common SE(3) diagnostic only"}); return report

def real_native_agreement(reference: np.ndarray, prediction: np.ndarray, myocardium_mask: np.ndarray) -> dict:
    report=_metrics(reference,prediction,np.asarray(myocardium_mask,bool)&np.isfinite(reference)&np.isfinite(prediction)); report["endpoint"]="native-reference agreement/consistency, not accuracy versus ground truth"; return report

def real_native_paired_agreement(reference: np.ndarray, proposed: np.ndarray, fit_first: np.ndarray, myocardium_mask: np.ndarray, proposed_valid: np.ndarray, fit_first_valid: np.ndarray) -> dict:
    """R0 pair metrics on one common SAX-myocardium support, never private supports."""
    reference, proposed, fit_first = (np.asarray(value, float) for value in (reference, proposed, fit_first))
    myocardium_mask, proposed_valid, fit_first_valid = (np.asarray(value, bool) for value in (myocardium_mask, proposed_valid, fit_first_valid))
    if not (reference.shape == proposed.shape == fit_first.shape == myocardium_mask.shape == proposed_valid.shape == fit_first_valid.shape): raise ValueError("R0 paired arrays must share one plane shape")
    common = myocardium_mask & np.isfinite(reference) & proposed_valid & fit_first_valid & np.isfinite(proposed) & np.isfinite(fit_first)
    if not common.any(): raise ValueError("empty common R0 SAX myocardium support")
    result = {"endpoint":"native-reference agreement/consistency, not accuracy versus ground truth","primary_region":"common_sax_myocardium","support_equation":"SAX_myocardium AND finite(reference) AND proposed_reprojection_valid AND fit_first_reprojection_valid AND finite(proposed, fit_first)","common_support_voxels":int(common.sum()),"proposed":_metrics(reference, proposed, common),"fit_first":_metrics(reference, fit_first, common)}
    return result

def paired_per_plane_records(rows: list[dict], method: str) -> list[dict]:
    """Expand common-support plane records safely, retaining zero-support status rows."""
    if method not in ("proposed", "fit_first"): raise ValueError(f"unknown paired method: {method}")
    records = []
    for row in rows:
        base = {key: value for key, value in row.items() if key not in ("proposed", "fit_first")}
        records.append({**base, **row[method]} if method in row else base)
    return records

def heldout_quantitative_metrics(reference: np.ndarray, proposed: np.ndarray, fit_first: np.ndarray, source_support: np.ndarray, reprojection_valid: np.ndarray) -> dict:
    """Evaluate both methods on one immutable, common held-out-plane support.

    The support is derived from native source masks/dictionary validity and
    geometry, then intersected with both prediction finiteness gates.  Thus a
    non-finite value is excluded equally from both methods and neither method
    receives a private support mask.
    """
    reference, proposed, fit_first = (np.asarray(value, float) for value in (reference, proposed, fit_first))
    source_support, reprojection_valid = (np.asarray(value, bool) for value in (source_support, reprojection_valid))
    if not (reference.shape == proposed.shape == fit_first.shape == source_support.shape == reprojection_valid.shape):
        raise ValueError("held-out arrays must share one plane shape")
    common = source_support & reprojection_valid & np.isfinite(reference) & np.isfinite(proposed) & np.isfinite(fit_first)
    if not common.any():
        raise ValueError("empty common held-out quantitative support")
    result = {"endpoint":"held-out native-plane quantitative agreement/generalization, not true 3-D accuracy","primary_region":"heldout_full_valid_slice","support_equation":"source_foreground AND native_dictionary_finite AND common_world_reprojection_valid AND finite(reference, proposed, fit_first)","common_support_voxels":int(common.sum()),"source_support_voxels":int(source_support.sum()),"proposed":_metrics(reference,proposed,common),"fit_first":_metrics(reference,fit_first,common)}
    for value in result.values():
        if isinstance(value, dict): value["coverage_fraction"] = float(common.sum() / source_support.sum()) if source_support.any() else 0.0
    return result

def main() -> None:
    import argparse, json, nibabel as nib
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--mode', choices=('synthetic','real','real-paired'), required=True); parser.add_argument('--synthetic-t1'); parser.add_argument('--synthetic-t2'); parser.add_argument('--gt-maps'); parser.add_argument('--subject'); parser.add_argument('--real-t1'); parser.add_argument('--real-t2'); parser.add_argument('--proposed-t1'); parser.add_argument('--proposed-t2'); parser.add_argument('--fit-first-t1'); parser.add_argument('--fit-first-t2'); parser.add_argument('--native-reference'); parser.add_argument('--myocardium-bundle'); parser.add_argument('--output', required=True); args=parser.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError(f'refusing overwrite: {output}')
    if args.mode == 'real-paired':
        if not all((args.subject,args.proposed_t1,args.proposed_t2,args.fit_first_t1,args.fit_first_t2,args.native_reference,args.myocardium_bundle)): raise ValueError('paired real evaluation requires both methods, references, and masks')
        ref_root=Path(args.native_reference); reference=np.load(ref_root/'sax.npz'); masks=np.load(Path(args.myocardium_bundle)/'sax_myocardium_masks.npz'); methods={'proposed':{},'fit_first':{}}
        volumes={'proposed':{'t1_ms':nib.load(args.proposed_t1),'t2_ms':nib.load(args.proposed_t2)},'fit_first':{'t1_ms':nib.load(args.fit_first_t1),'t2_ms':nib.load(args.fit_first_t2)}}
        for parameter in ('t1_ms','t2_ms'):
            native=nib.load(str(ref_root/('sax_T1_stack_ms.nii.gz' if parameter=='t1_ms' else 'sax_T2_stack_ms.nii.gz'))); rows=[]; aggregate_reference=[]; aggregate_predictions={'proposed':[],'fit_first':[]}; total_support=0; common_support=0
            for group in range(reference[parameter].shape[0]):
                proposed,pvalid=reproject_volume_to_native_plane(volumes['proposed'][parameter].get_fdata(),volumes['proposed'][parameter].affine,native.affine,reference[parameter].shape[1:],group); baseline,bvalid=reproject_volume_to_native_plane(volumes['fit_first'][parameter].get_fdata(),volumes['fit_first'][parameter].affine,native.affine,reference[parameter].shape[1:],group); support=masks['myocardium_full'][group].astype(bool); total_support+=int(support.sum()); common=support & np.isfinite(reference[parameter][group]) & pvalid & bvalid & np.isfinite(proposed) & np.isfinite(baseline)
                if not common.any(): rows.append({'group_index':int(group),'status':'no_common_myocardium_support','support_voxels':0,'coverage_fraction':0.0}); continue
                pair=real_native_paired_agreement(reference[parameter][group],proposed,baseline,support,pvalid,bvalid); common_support+=int(common.sum()); pair.update({'group_index':int(group),'support_pixel_count':int(support.sum()),'coverage_fraction':float(common.sum()/support.sum())}); rows.append(pair); aggregate_reference.append(reference[parameter][group][common]); aggregate_predictions['proposed'].append(proposed[common]); aggregate_predictions['fit_first'].append(baseline[common])
            if not aggregate_reference: raise ValueError(f'{args.subject}/{parameter} has no common SAX myocardium support')
            for method in ('proposed','fit_first'):
                aggregate=_metrics(np.concatenate(aggregate_reference),np.concatenate(aggregate_predictions[method]),np.ones(sum(map(len,aggregate_reference)),bool)); aggregate.update({'endpoint':'native-reference agreement/consistency, not accuracy versus ground truth','primary_region':'common_sax_myocardium','support_equation':'SAX_myocardium AND finite(reference) AND proposed_reprojection_valid AND fit_first_reprojection_valid AND finite(proposed, fit_first)','statistical_unit':'subject; common valid SAX myocardium pixels aggregated; pixels are not inferential samples','coverage_fraction':common_support/total_support,'common_support_count':common_support,'total_myocardium_support_count':total_support}); methods[method][parameter]={'subject_level':aggregate,'per_native_sax_plane':paired_per_plane_records(rows, method)}
        output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps({'subject':args.subject,'endpoint':'agreement/consistency, not 3-D ground-truth accuracy','pairwise_common_support':True,'methods':methods},indent=2)); print(output); return
    if args.mode == 'real':
        if not all((args.subject,args.real_t1,args.real_t2,args.native_reference,args.myocardium_bundle)): raise ValueError('real evaluation requires subject, T1/T2 volumes, current reference, and masks')
        ref_root=Path(args.native_reference); reference=np.load(ref_root/'sax.npz'); masks=np.load(Path(args.myocardium_bundle)/'sax_myocardium_masks.npz'); result={}
        for parameter,volume_path in (('t1_ms',args.real_t1),('t2_ms',args.real_t2)):
            image=nib.load(volume_path); native=nib.load(str(ref_root/('sax_T1_stack_ms.nii.gz' if parameter=='t1_ms' else 'sax_T2_stack_ms.nii.gz'))); rows=[]; aggregate_reference=[]; aggregate_prediction=[]; total_support=0; valid_support=0
            for group in range(reference[parameter].shape[0]):
                predicted, valid = reproject_volume_to_native_plane(image.get_fdata(),image.affine,native.affine,reference[parameter].shape[1:],group)
                support=masks['myocardium_full'][group].astype(bool); total_support += int(support.sum()); valid_plane=support & valid & np.isfinite(reference[parameter][group]) & np.isfinite(predicted); valid_support += int(valid_plane.sum())
                if valid_plane.any():
                    row=real_native_agreement(reference[parameter][group],predicted,valid_plane); row['group_index']=int(group); row['support_pixel_count']=int(support.sum()); row['coverage_fraction']=float(valid_plane.sum()/support.sum()); rows.append(row)
                    aggregate_reference.append(reference[parameter][group][valid_plane]); aggregate_prediction.append(predicted[valid_plane])
                else: rows.append({'group_index':int(group),'status':'no_valid_myocardium_support','support_voxels':0,'coverage_fraction':0.0})
            if not aggregate_reference: raise ValueError(f'{args.subject}/{parameter} has no valid myocardium support')
            aggregate=_metrics(np.concatenate(aggregate_reference),np.concatenate(aggregate_prediction),np.ones(sum(map(len,aggregate_reference)),bool)); aggregate.update({'endpoint':'native-reference agreement/consistency, not accuracy versus ground truth','statistical_unit':'subject; all valid SAX myocardium pixels aggregated; pixels are not inferential samples','coverage_fraction':valid_support/total_support,'valid_support_count':valid_support,'total_myocardium_support_count':total_support})
            result[parameter]={'subject_level':aggregate,'per_native_sax_plane':rows}
        output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps({'subject':args.subject,'endpoint':'agreement/consistency, not 3-D ground-truth accuracy','metrics':result},indent=2)); print(output); return
    if not all((args.synthetic_t1,args.synthetic_t2,args.gt_maps)): raise ValueError('synthetic evaluation requires T1/T2 NIfTI and Phase-1.2 GT maps')
    gt=np.load(args.gt_maps); affine=np.eye(4); affine[:3,:3]=np.diag(gt['spacing_mm']); affine[:3,3]=gt['origin_mm']
    result={}
    for parameter,path,normal in (('t1_ms',args.synthetic_t1,1410.0),('t2_ms',args.synthetic_t2,50.0)):
        image=nib.load(path); prediction,valid=resample_volume_to_grid(image.get_fdata(),image.affine,gt[parameter].shape,affine); result[parameter]=synthetic_metrics(gt[parameter],prediction,gt['myocardium_mask'],valid,normal)
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps({'endpoint':'synthetic XCAT3D myocardium; primary NRMSE uses fixed GT value','metrics':result},indent=2)); print(output)

if __name__ == '__main__': main()
