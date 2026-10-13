"""Versioned current-protocol native-reference creator for Phase 2 paired real data."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib

from .contracts import historical_reference_is_current_compatible, sha256, write_immutable_json
from .fit_first import fit_native

def build_current_native_reference(observations: str|Path, output: str|Path, *, legacy_config: dict | None = None) -> Path:
    """Generate effective (not 3-D-GT) maps with the frozen current dictionary matcher."""
    source, destination = Path(observations), Path(output)
    if destination.exists(): raise FileExistsError(f'refusing overwrite: {destination}')
    destination.mkdir(parents=True)
    maps_root = fit_native(source, destination/'dictionary_fit')
    maps, geometry = np.load(maps_root/'native_maps.npz'), np.load(maps_root/'geometry.npz')
    affine_lps = np.asarray(geometry['affine_lps_rc'][0], float).copy()
    if len(geometry['affine_lps_rc']) > 1:
        affine_lps[:3, 2] = np.asarray(geometry['affine_lps_rc'][1], float)[:3, 3] - affine_lps[:3, 3]
    affine_ras = np.diag([-1., -1., 1., 1.]) @ affine_lps
    arrays = {key: np.asarray(maps[key]) for key in ('t1_ms','t2_ms','b1','group_idx')}
    arrays['valid_mask'] = np.isfinite(arrays['t1_ms']) & np.isfinite(arrays['t2_ms'])
    np.savez_compressed(destination/'sax.npz', **arrays)
    for parameter in ('t1','t2'):
        nib.save(nib.Nifti1Image(np.moveaxis(arrays[f'{parameter}_ms'], 0, 2).astype(np.float32), affine_ras), str(destination/f'sax_{parameter.upper()}_stack_ms.nii.gz'))
    legacy_status = 'not_supplied' if legacy_config is None else ('compatible_but_not_migrated' if historical_reference_is_current_compatible(legacy_config) else 'historical_protocol_rejected_TI_10_100')
    write_immutable_json(destination/'native_reference_manifest.json', {'role':'effective native 2-D agreement reference, not 3-D ground truth; dictionary-based references favor fit-first','current_protocol_source_observations':{'path':str(source.resolve()),'sha256':sha256(source)},'dictionary_maps':{'path':str((maps_root/'native_maps.npz').resolve()),'sha256':sha256(maps_root/'native_maps.npz')},'legacy_reference_status':legacy_status,'affine_ras':affine_ras.tolist(),'hashes':{name:sha256(destination/name) for name in ('sax.npz','sax_T1_stack_ms.nii.gz','sax_T2_stack_ms.nii.gz')},'interpolation':'none'})
    return destination

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--observations',required=True); parser.add_argument('--output',required=True); args=parser.parse_args(); print(build_current_native_reference(**vars(args)))
if __name__ == '__main__': main()
