"""Audit and immutably migrate pre-7a84b33 R0 native dictionary maps without refitting."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .contracts import BASELINE_POSE_POLICY, sha256, write_immutable_json
from .fit_first import NATIVE_FIT_METHOD, STACKS, _complete_source_group, _dictionary_contract, _native_group_audit


def _migrate_stack(old_root: Path, prepared_root: Path, target: Path, stack: str, expected_observation_sha256: str | None) -> None:
    source = prepared_root / stack / 'observations.npz'
    old_stack, new_stack = old_root / stack, target / stack
    maps_path, geometry_path = old_stack / 'native_maps.npz', old_stack / 'geometry.npz'
    if not source.is_file() or not maps_path.is_file() or not geometry_path.is_file():
        raise FileNotFoundError(f'{stack}: migration requires prepared observations plus native maps and geometry')
    if expected_observation_sha256 is not None and sha256(source) != expected_observation_sha256:
        raise ValueError(f'{stack}: prepared observation SHA256 does not match historical R0 provenance')
    with np.load(source, allow_pickle=False) as observations, np.load(maps_path, allow_pickle=False) as maps, np.load(geometry_path, allow_pickle=False) as geometry:
        for name in ('t1_ms', 't2_ms', 'b1', 'group_idx'):
            if name not in maps.files:
                raise ValueError(f'{stack}: native maps missing {name}')
        if any(name not in geometry.files for name in ('affine_lps_rc', 'pixel_spacing_rc_mm', 'slice_thickness_mm')):
            raise ValueError(f'{stack}: native geometry is incomplete')
        groups = [int(value) for value in maps['group_idx']]
        source_groups = [int(value) for value in np.unique(observations['group_idx'])]
        if len(groups) != len(set(groups)) or groups != source_groups:
            raise ValueError(f'{stack}: native-map group IDs must uniquely match full prepared observations')
        if any(array.shape[0] != len(groups) for array in (maps['t1_ms'], maps['t2_ms'], maps['b1'], geometry['affine_lps_rc'], geometry['pixel_spacing_rc_mm'], geometry['slice_thickness_mm'])):
            raise ValueError(f'{stack}: native map/geometry group dimensions disagree')
        for group in groups:
            _complete_source_group(observations, group, source)
        group_audit = _native_group_audit(observations, np.asarray(groups, dtype=observations['group_idx'].dtype))
    new_stack.mkdir()
    shutil.copy2(maps_path, new_stack / 'native_maps.npz')
    shutil.copy2(geometry_path, new_stack / 'geometry.npz')
    if sha256(maps_path) != sha256(new_stack / 'native_maps.npz') or sha256(geometry_path) != sha256(new_stack / 'geometry.npz'):
        raise ValueError(f'{stack}: copied native-map SHA256 verification failed')
    write_immutable_json(new_stack / 'manifest.json', {
        'method': NATIVE_FIT_METHOD,
        'source_observations': {'path': str(source.resolve()), 'sha256': sha256(source)},
        'dictionary_contract': _dictionary_contract(),
        'group_ids': groups,
        'group_audit': group_audit,
        'native_maps_sha256': sha256(new_stack / 'native_maps.npz'),
        'geometry_sha256': sha256(new_stack / 'geometry.npz'),
        'code_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'migration_source_path': str(old_stack.resolve()),
        'migration_source_sha256': sha256(maps_path),
        'migration_source_native_maps_sha256': sha256(maps_path),
        'migration_source_geometry_sha256': sha256(geometry_path),
        'migration_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'numerical_fitting_rerun': False,
        'migration_statement': 'Audited copy of pre-7a84b33 native maps; numerical fitting was NOT rerun.',
        'baseline_pose_policy': BASELINE_POSE_POLICY,
        'next_formal_step': 'independent T1 and T2 NeSVoR stack registration/transformation optimization',
    })


def migrate_native_fit_cache(old_native_maps_root: str | Path, prepared_root: str | Path, output: str | Path) -> Path:
    """Migrate legacy R0 native maps into a new auditable cache without calling ``fit_native``."""
    old_root, prepared, target = Path(old_native_maps_root), Path(prepared_root), Path(output)
    if target.exists():
        raise FileExistsError(f'refusing overwrite: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.staging-', dir=target.parent))
    try:
        legacy_input_manifest = old_root.parent / 'input_manifest.json'
        historical_observations = json.loads(legacy_input_manifest.read_text()).get('observations', {}) if legacy_input_manifest.is_file() else {}
        for stack in STACKS:
            _migrate_stack(old_root, prepared, staging, stack, historical_observations.get(stack))
        if target.exists():
            raise FileExistsError(f'refusing overwrite: {target}')
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-native-maps-root', required=True)
    parser.add_argument('--prepared-root', required=True, help='Formal full prepared root for one subject')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(migrate_native_fit_cache(args.old_native_maps_root, args.prepared_root, args.output))


if __name__ == '__main__':
    main()
