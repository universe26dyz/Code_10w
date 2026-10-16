"""Native current-protocol dictionary maps for the independent-NeSVoR baseline."""
from __future__ import annotations
import argparse, json, shlex, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from .contracts import BASELINE_POSE_POLICY, FROZEN_GRID, HISTORICAL_NESVOR_EXECUTABLE_DEFAULT, sha256, write_immutable_json
from .dictionary import frozen_dictionary, match_fingerprints

STACKS = ('sax', '2ch', '4ch')
NATIVE_FIT_METHOD = 'current-protocol HHZ/TRAD dictionary matching; normalized 10D inner product; no interpolation/refinement'


def _dictionary_contract() -> dict[str, object]:
    return {
        'grid': {name: list(values) if isinstance(values, tuple) else values for name, values in FROZEN_GRID.items()},
        'matching': 'normalized 10D cosine/inner-product matching',
        'interpolation_or_refinement': False,
        'protocol': {'tr_ms': 2.61, 'vps': 87},
    }


def _native_group_audit(data: np.lib.npyio.NpzFile, groups: np.ndarray) -> list[dict[str, object]]:
    result = []
    for group in groups:
        rows = np.flatnonzero(data['group_idx'] == group)
        result.append({'group_idx': int(group), 'weight_idx': [int(value) for value in np.sort(data['weight_idx'][rows])], 'timing9_ms': [float(value) for value in data['timing9_ms'][rows[0]]], 'tr_ms': float(data['tr_ms'][rows[0]]), 'vps': int(data['vps'][rows[0]])})
    return result

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
    write_immutable_json(output/'manifest.json', {'method': NATIVE_FIT_METHOD, 'source_observations': {'path': str(source.resolve()), 'sha256': sha256(source)}, 'dictionary_contract': _dictionary_contract(), 'group_ids': [int(group) for group in groups], 'group_audit': _native_group_audit(data, groups), 'native_maps_sha256': sha256(output/'native_maps.npz'), 'geometry_sha256': sha256(output/'geometry.npz'), 'code_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'dictionary_cache_entries': len(cache), 'baseline_pose_policy': BASELINE_POSE_POLICY, 'next_formal_step': 'independent T1 and T2 NeSVoR stack registration/transformation optimization'})
    return output


def _complete_source_group(data: np.lib.npyio.NpzFile, group: int, source: Path) -> None:
    rows = np.flatnonzero(data['group_idx'] == group)
    if rows.size != 10 or not np.array_equal(np.sort(data['weight_idx'][rows]), np.arange(10)):
        raise ValueError(f'{source}: cached group {group} must contain exactly weights 0..9')
    timing = np.asarray(data['timing9_ms'][rows[0]], float)
    if any(not np.array_equal(data['timing9_ms'][row], timing) or float(data['tr_ms'][row]) != 2.61 or int(data['vps'][row]) != 87 for row in rows):
        raise ValueError(f'{source}: cached group {group} violates frozen timing/TR/VPS')


def reuse_native_maps(full_native_maps_root: str | Path, sparse_manifest: str | Path, output: str | Path) -> Path:
    """Subset audited R0 native maps for one eligible R1 fold without refitting."""
    target = Path(output)
    if target.exists():
        raise FileExistsError(f'refusing overwrite: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.staging-', dir=target.parent))
    try:
        _reuse_native_maps_into(Path(full_native_maps_root), Path(sparse_manifest), staging)
        if target.exists():
            raise FileExistsError(f'refusing overwrite: {target}')
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def _reuse_native_maps_into(source_root: Path, manifest_path: Path, target: Path) -> None:
    sparse = json.loads(manifest_path.read_text())
    if sparse.get('status') != 'ELIGIBLE':
        raise ValueError('native-map reuse requires an ELIGIBLE sparse manifest')
    if sparse.get('condition') == 'L100':
        raise ValueError('L100 reuses R0 outputs and must not create a sparse reconstruction')
    if set(sparse.get('stacks', {})) != set(STACKS):
        raise ValueError('sparse manifest stack names must be exactly SAX/2CH/4CH')
    records: dict[str, object] = {}
    for stack in STACKS:
        sparse_stack = sparse['stacks'][stack]
        retained = [int(value) for value in sparse_stack.get('retained_original_group_ids', [])]
        heldout = [int(value) for value in sparse_stack.get('heldout_original_group_ids', [])]
        if len(retained) != len(set(retained)):
            raise ValueError(f'{stack}: retained original group IDs must be unique')
        cached_root = source_root / stack
        maps_path, geometry_path, cached_manifest_path = cached_root/'native_maps.npz', cached_root/'geometry.npz', cached_root/'manifest.json'
        if not all(path.is_file() for path in (maps_path, geometry_path, cached_manifest_path)):
            raise FileNotFoundError(f'{stack}: full-data native-map artifact is incomplete')
        cached = json.loads(cached_manifest_path.read_text())
        if cached.get('method') != NATIVE_FIT_METHOD or cached.get('dictionary_contract') != _dictionary_contract():
            raise ValueError(f'{stack}: cached native maps do not satisfy the frozen dictionary contract')
        if cached.get('native_maps_sha256') != sha256(maps_path) or cached.get('geometry_sha256') != sha256(geometry_path):
            raise ValueError(f'{stack}: cached native-map artifact SHA256 mismatch')
        source_observation = cached.get('source_observations', {})
        sparse_observation = sparse_stack.get('source_observation', {})
        if source_observation.get('sha256') != sparse_observation.get('sha256'):
            raise ValueError(f'{stack}: source observation SHA256 mismatch')
        source_path = Path(str(source_observation.get('path', '')))
        if not source_path.is_file() or sha256(source_path) != source_observation.get('sha256'):
            raise ValueError(f'{stack}: cached source observation identity is unavailable or mismatched')
        with np.load(maps_path, allow_pickle=False) as maps, np.load(geometry_path, allow_pickle=False) as geometry, np.load(source_path, allow_pickle=False) as observations:
            for name in ('t1_ms', 't2_ms', 'b1', 'group_idx'):
                if name not in maps.files:
                    raise ValueError(f'{stack}: cached native maps missing {name}')
            if any(name not in geometry.files for name in ('affine_lps_rc', 'pixel_spacing_rc_mm', 'slice_thickness_mm')):
                raise ValueError(f'{stack}: cached geometry is incomplete')
            cached_groups = [int(value) for value in maps['group_idx']]
            if len(cached_groups) != len(set(cached_groups)):
                raise ValueError(f'{stack}: each cached original group must exist exactly once')
            if any(array.shape[0] != len(cached_groups) for array in (maps['t1_ms'], maps['t2_ms'], maps['b1'], geometry['affine_lps_rc'], geometry['pixel_spacing_rc_mm'], geometry['slice_thickness_mm'])):
                raise ValueError(f'{stack}: cached map/geometry group dimensions disagree')
            if stack == 'sax' and (retained != cached_groups or heldout):
                raise ValueError('sax: R1 reuse requires every full-data SAX group and no held-out SAX group')
            if set(retained).intersection(heldout):
                raise ValueError(f'{stack}: retained and held-out original groups overlap')
            positions = []
            for group in retained:
                if cached_groups.count(group) != 1:
                    raise ValueError(f'{stack}: retained original group {group} must exist exactly once in cached maps')
                _complete_source_group(observations, group, source_path)
                positions.append(cached_groups.index(group))
            stack_target = target / stack
            stack_target.mkdir()
            if stack == 'sax':
                shutil.copy2(maps_path, stack_target/'native_maps.npz')
                shutil.copy2(geometry_path, stack_target/'geometry.npz')
            else:
                np.savez_compressed(stack_target/'native_maps.npz', t1_ms=maps['t1_ms'][positions], t2_ms=maps['t2_ms'][positions], b1=maps['b1'][positions], group_idx=np.asarray(retained, dtype=maps['group_idx'].dtype))
                np.savez_compressed(stack_target/'geometry.npz', affine_lps_rc=geometry['affine_lps_rc'][positions], pixel_spacing_rc_mm=geometry['pixel_spacing_rc_mm'][positions], slice_thickness_mm=geometry['slice_thickness_mm'][positions])
        records[stack] = {'retained_original_group_ids': retained, 'heldout_original_group_ids': heldout, 'nesvor_input_original_group_ids': retained, 'reused_source_map_sha256': sha256(maps_path), 'reused_source_geometry_sha256': sha256(geometry_path), 'subset_output_map_sha256': sha256(stack_target/'native_maps.npz'), 'subset_output_geometry_sha256': sha256(stack_target/'geometry.npz')}
    write_immutable_json(target/'reuse_manifest.json', {'method': 'fit_first_native_map_group_local_reuse', 'sparse_manifest': {'path': str(manifest_path.resolve()), 'sha256': sha256(manifest_path)}, 'full_native_maps_root': str(source_root.resolve()), 'native_fit_reuse_is_group_local': True, 'heldout_groups_never_passed_to_nesvor': True, 'stacks': records})

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
    export_provenance = {}
    for parameter in ('t1', 't2'):
        raw_values = np.asarray(maps[f'{parameter}_ms'], np.float32) / float((normalization or {}).get(parameter, 1.0))
        invalid = ~np.isfinite(raw_values)
        values = np.where(invalid, 0.0, raw_values).astype(np.float32)
        if not np.isfinite(values).all(): raise ValueError(f'non-finite NeSVoR export for {parameter}')
        folder = target/parameter; folder.mkdir()
        nib.save(nib.Nifti1Image(np.moveaxis(values, 0, 2), affine_ras), str(folder/'stack.nii.gz'))
        export_provenance[parameter] = {'nonfinite_native_pixel_count': int(invalid.sum()), 'finite_output': True}
    write_immutable_json(target/'manifest.json', {'input_native_maps': {'path': str(source.resolve()), 'sha256': sha256(source/'native_maps.npz')}, 'group_idx': [int(x) for x in maps['group_idx']], 'affine_ras': affine_ras.tolist(), 'geometry': 'current prepared LPS row/column geometry converted to RAS; groups become stack slices', 'normalization_before_nesvor': normalization or {'t1':1.0,'t2':1.0}, 'denormalization_after_nesvor_ms': normalization or {'t1':1.0,'t2':1.0}, 'nesvor_background_policy': 'non-finite native dictionary pixels are exported as zero-valued background; finite matched values unchanged', 'nesvor_export': export_provenance, 'pose_policy': BASELINE_POSE_POLICY, 'parameters': ['t1', 't2']})
    return target

def build_independent_nesvor_commands(nesvor_input: str|Path|list[str|Path], output: str|Path, device: int = 0, nesvor_executable: str|Path = HISTORICAL_NESVOR_EXECUTABLE_DEFAULT) -> list[str]:
    """Exact independent NeSVoR commands; neither command may consume the other's pose."""
    roots = [Path(value) for value in nesvor_input] if isinstance(nesvor_input, list) else [Path(nesvor_input)]
    result = Path(output)
    if result.exists(): raise FileExistsError(f'refusing overwrite: {result}')
    commands=[]
    for parameter in ('t1', 't2'):
        destination = result/parameter
        inputs=[str(root/parameter/'stack.nii.gz') for root in roots]; scale='2500' if parameter=='t1' else '200'
        args = [str(nesvor_executable),'reconstruct','--input-stacks',*inputs,'--thicknesses','8','8','8','--registration','stack','--scanner-space','--metric','none','--weight-transformation','1','--weight-deform','0.1','--weight-image','1','--weight-pixel-var-smooth','0','--output-volume',str(destination/'volume_normalized.nii.gz'),'--output-model',str(destination/'model.pt'),'--output-json',str(destination/'result.json'),'--output-log',str(destination/'nesvor.log'),'--output-slices',str(destination/'slices'),'--output-intensity-mean','0','--output-resolution','1.0','--output-psf-factor','1','--no-slice-scale','--single-precision','--log2-hashmap-size','19','--batch-size','2000','--n-iter','10000','--n-samples','256','--n-inference-samples','128','--inference-batch-size','16000','--pixel-variance-weight','1','--coarsest-resolution','16','--finest-resolution','0.5','--level-scale','1.3819','--n-features-per-level','2','--depth','1','--width','64','--n-features-z','15','--n-features-slice','16','--weight-bias','100','--image-regularization','edge','--delta','0.2','--learning-rate','0.005','--gamma','0.33','--milestones','0.5','0.75','0.9','--seed','20260823','--device',str(device)]
        commands.append(' '.join(shlex.quote(value) for value in args)+f' && {shlex.quote(sys.executable)} -m paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_denormalize --input {shlex.quote(str(destination/"volume_normalized.nii.gz"))} --scale {scale} --output {shlex.quote(str(destination/"volume_ms.nii.gz"))}')
    return commands
def main()->None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--observations',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(fit_native(**vars(a)))
if __name__=='__main__':main()
