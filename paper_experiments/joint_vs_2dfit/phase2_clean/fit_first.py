"""Native current-protocol dictionary maps for the independent-NeSVoR baseline."""
from __future__ import annotations
import argparse, json, shlex
from pathlib import Path
import numpy as np
from .contracts import BASELINE_POSE_POLICY, sha256, write_immutable_json
from .dictionary import frozen_dictionary, match_fingerprints

def fit_native(observations: str|Path, output: str|Path) -> Path:
    source,output=Path(observations),Path(output)
    if output.exists(): raise FileExistsError(f'refusing overwrite: {output}')
    data=np.load(source); groups=np.unique(data['group_idx']); h,w=data['images'].shape[-2:]
    t1=np.full((len(groups),h,w),np.nan,np.float32);t2=t1.copy();b1=t1.copy(); cache={}
    for out_group,group in enumerate(groups):
        rows=np.flatnonzero(data['group_idx']==group)
        if rows.size!=10 or not np.array_equal(np.sort(data['weight_idx'][rows]),np.arange(10)): raise ValueError('each group must include weights 0..9')
        ordered=rows[np.argsort(data['weight_idx'][rows])]; timing=tuple(np.asarray(data['timing9_ms'][ordered[0]],float)); tr=float(data['tr_ms'][ordered[0]]); vps=int(data['vps'][ordered[0]])
        if any(not np.array_equal(data['timing9_ms'][x],timing) or float(data['tr_ms'][x])!=tr or int(data['vps'][x])!=vps for x in ordered): raise ValueError('group timing/TR/VPS must be shared')
        key=(timing,tr,vps)
        if key not in cache: cache[key]=frozen_dictionary(np.asarray(timing),tr,vps)
        valid=np.all(data['masks'][ordered].astype(bool),axis=0)&np.isfinite(data['images'][ordered]).all(axis=0)
        idx=match_fingerprints(cache[key]['fingerprints'],np.moveaxis(data['images'][ordered,:, :],0,-1)[valid])
        for target,name in ((t1,'t1_ms'),(t2,'t2_ms'),(b1,'b1')): target[out_group][valid]=cache[key][name][idx]
    output.mkdir(parents=True); np.savez_compressed(output/'native_maps.npz',t1_ms=t1,t2_ms=t2,b1=b1,group_idx=groups)
    first_rows = [np.flatnonzero(data['group_idx'] == group)[0] for group in groups]
    np.savez_compressed(output/'geometry.npz', affine_lps_rc=data['affine_lps_rc'][first_rows], pixel_spacing_rc_mm=data['pixel_spacing_rc_mm'][first_rows], slice_thickness_mm=data['slice_thickness_mm'][first_rows])
    write_immutable_json(output/'manifest.json',{"method":"current-protocol HHZ/TRAD dictionary matching; normalized 10D inner product; no interpolation/refinement","source_observations":{"path":str(source.resolve()),"sha256":sha256(source)},"dictionary_cache_entries":len(cache),"baseline_pose_policy":BASELINE_POSE_POLICY,"next_formal_step":"independent T1 and T2 NeSVoR stack registration/transformation optimization"})
    return output

def export_nesvor_native_inputs(native_maps: str|Path, output: str|Path, normalization: dict[str, float] | None = None) -> Path:
    """Export current dictionary maps as independent geometry-preserving NeSVoR stacks."""
    import nibabel as nib
    source, target = Path(native_maps), Path(output)
    if target.exists(): raise FileExistsError(f'refusing overwrite: {target}')
    maps, geometry = np.load(source/'native_maps.npz'), np.load(source/'geometry.npz')
    affine_lps = np.asarray(geometry['affine_lps_rc'][0], float).copy()
    if len(geometry['affine_lps_rc']) > 1:
        affine_lps[:3, 2] = np.asarray(geometry['affine_lps_rc'][1], float)[:3, 3] - affine_lps[:3, 3]
    affine_ras = np.diag([-1., -1., 1., 1.]) @ affine_lps
    target.mkdir(parents=True)
    for parameter in ('t1', 't2'):
        values = np.asarray(maps[f'{parameter}_ms'], np.float32) / float((normalization or {}).get(parameter, 1.0))
        folder = target/parameter; folder.mkdir()
        nib.save(nib.Nifti1Image(np.moveaxis(values, 0, 2), affine_ras), str(folder/'stack.nii.gz'))
    write_immutable_json(target/'manifest.json', {'input_native_maps': {'path': str(source.resolve()), 'sha256': sha256(source/'native_maps.npz')}, 'group_idx': [int(x) for x in maps['group_idx']], 'affine_ras': affine_ras.tolist(), 'geometry': 'current prepared LPS row/column geometry converted to RAS; groups become stack slices', 'normalization_before_nesvor': normalization or {'t1':1.0,'t2':1.0}, 'denormalization_after_nesvor_ms': normalization or {'t1':1.0,'t2':1.0}, 'pose_policy': BASELINE_POSE_POLICY, 'parameters': ['t1', 't2']})
    return target

def build_independent_nesvor_commands(nesvor_input: str|Path, output: str|Path, device: int = 0) -> list[str]:
    """Exact independent NeSVoR commands; neither command may consume the other's pose."""
    root, result = Path(nesvor_input), Path(output)
    if result.exists(): raise FileExistsError(f'refusing overwrite: {result}')
    commands=[]
    for parameter in ('t1', 't2'):
        destination = result/parameter
        args = ['nesvor','reconstruct','--input-stacks',str(root/parameter/'stack.nii.gz'),'--registration','stack','--output-volume',str(destination/'volume.nii.gz'),'--output-json',str(destination/'result.json'),'--output-log',str(destination/'nesvor.log'),'--output-slices',str(destination/'slices'),'--output-resolution','1.0','--no-slice-scale','--n-iter','10000','--n-samples','256','--inference-samples','128','--pixel-variance-weight','1.0','--device',str(device)]
        commands.append(' '.join(shlex.quote(value) for value in args))
    return commands
def main()->None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--observations',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(fit_native(**vars(a)))
if __name__=='__main__':main()
