"""Formal fit-first driver: dictionary maps then independent T1/T2 NeSVoR executions."""
from __future__ import annotations
import argparse, json, subprocess, sys
import shlex
from pathlib import Path
from .contracts import HISTORICAL_NESVOR_EXECUTABLE_DEFAULT, sha256, write_immutable_json
from .fit_first import fit_native, reuse_native_maps, export_nesvor_native_inputs, build_independent_nesvor_commands
from .fit_first_intensity_preserving import _selected_standardized_maps
from .reuse_standardized_native_dictionary_maps import validate_standardized_native_dictionary_reuse
from .sparse_profile_manifest import SCHEMA_VERSION, validate_sparse_profile_manifest

def build_baseline_run(prepared_root: str|Path, subject_id: str, output: str|Path, nesvor_executable: str|Path = HISTORICAL_NESVOR_EXECUTABLE_DEFAULT, *, reuse_native_maps_root: str|Path|None = None, sparse_manifest: str|Path|None = None, standardized_native_maps_root: str|Path|None = None, standardized_reuse_manifest: str|Path|None = None, expected_generator_git_sha: str|None = None, expected_manifest_sha256: str|None = None, device: str|int = 0) -> tuple[Path, list[str]]:
    base, destination = Path(prepared_root), Path(output)
    sparse_profile = validate_sparse_profile_manifest(sparse_manifest, subject=subject_id, strict=expected_generator_git_sha is not None, expected_generator_git_sha=expected_generator_git_sha, expected_manifest_sha256=expected_manifest_sha256) if sparse_manifest is not None else None
    prepared = base if sparse_profile is not None or all((base/stack/'observations.npz').is_file() for stack in ('sax','2ch','4ch')) else base/subject_id
    if destination.exists(): raise FileExistsError(f'refusing overwrite: {destination}')
    destination.mkdir(parents=True)
    standardized_mode = standardized_native_maps_root is not None or standardized_reuse_manifest is not None
    if not standardized_mode and (reuse_native_maps_root is None) != (sparse_manifest is None):
        raise ValueError('native-map reuse requires both reuse_native_maps_root and sparse_manifest')
    if (standardized_native_maps_root is None) != (standardized_reuse_manifest is None):
        raise ValueError('standardized native-map reuse requires both standardized_native_maps_root and standardized_reuse_manifest')
    if standardized_mode and reuse_native_maps_root is not None:
        raise ValueError('standardized native-map reuse is mutually exclusive with R0 sparse reuse')
    native_maps = destination/'native_maps'
    standardized_reuse = None
    standardized_observations = None
    selected_standardized_records = None
    if standardized_mode:
        standardized_reuse = validate_standardized_native_dictionary_reuse(standardized_reuse_manifest, subject=subject_id)
        standardized_reuse = {**standardized_reuse, '_manifest_path': str(Path(standardized_reuse_manifest).resolve())}
        supplied_root = Path(standardized_native_maps_root).resolve()
        manifest_root = Path(standardized_reuse_manifest).resolve().parent
        if supplied_root != manifest_root:
            raise ValueError('standardized native maps root must be the parent of the validated reuse manifest')
        native_maps, active_stacks, selected_standardized_records = _selected_standardized_maps(
            standardized_root=supplied_root, reuse=standardized_reuse, sparse=sparse_profile,
            output=destination / 'selected_standardized_native_maps',
        )
        standardization_payload = json.loads(Path(standardized_reuse['standardization_manifest']['path']).read_text())
        standardized_observations = {stack: standardization_payload['stacks'][stack]['output_observation']['sha256'] for stack in ('sax', '2ch', '4ch')}
        native_reuse = {'mode': 'standardized_native_dictionary_reuse', 'reuse_manifest': {'path': str(Path(standardized_reuse_manifest).resolve()), 'sha256': sha256(standardized_reuse_manifest)}, 'standardization_manifest': standardized_reuse['standardization_manifest'], 'dictionary_refit': False, 'target_counts': {stack: standardized_reuse['stacks'][stack]['target_count'] for stack in ('sax', '2ch', '4ch')}}
    elif reuse_native_maps_root is None:
        for stack in ('sax','2ch','4ch'):
            fit_native(prepared/stack/'observations.npz', native_maps/stack)
        native_reuse = None
    else:
        reuse_native_maps(reuse_native_maps_root, sparse_manifest, native_maps)
        native_reuse = {'full_native_maps_root': str(Path(reuse_native_maps_root).resolve()), 'sparse_manifest': {'path': str(Path(sparse_manifest).resolve()), 'sha256': sha256(sparse_manifest)}, 'reuse_manifest': {'path': str((native_maps/'reuse_manifest.json').resolve()), 'sha256': sha256(native_maps/'reuse_manifest.json')}}
    exports=[]
    reuse_records = json.loads((native_maps/'reuse_manifest.json').read_text())['stacks'] if reuse_native_maps_root is not None else {}
    if not standardized_mode:
        active_stacks = [stack for stack in ('sax','2ch','4ch') if native_reuse is None or reuse_records[stack]['training_stack_present']]
    if not active_stacks:
        raise ValueError('fit-first sparse profile has no training stacks eligible for NeSVoR export')
    for stack in active_stacks:
        maps=native_maps/stack
        exports.append(export_nesvor_native_inputs(maps, destination/'nesvor_inputs'/stack, {'t1':2500.0,'t2':200.0}))
    sample_masks = None
    raw_output_sampling_mask = None
    needs_explicit_sample_mask = standardized_mode or (sparse_profile is not None and sparse_profile.get('schema_version') == SCHEMA_VERSION)
    if needs_explicit_sample_mask:
        if 'sax' not in active_stacks:
            raise ValueError('formal Fit-first requires an active SAX training stack for deterministic output sampling')
        sample_masks = {parameter: destination/'nesvor_inputs'/'sax'/parameter/'stack.nii.gz' for parameter in ('t1', 't2')}
        for parameter, path in sample_masks.items():
            if not path.is_file(): raise FileNotFoundError(f'{parameter}: formal Fit-first SAX sample mask is missing: {path}')
        raw_output_sampling_mask = {'policy': 'explicit_parameter_matched_sax_native_support', 'scope': 'post_training_raw_volume_sampling_only', 'training_affected': False, 'registration_affected': False, 'pose_optimization_affected': False, 'formal_endpoint_affected': False, 'sample_masks': {parameter: {'path': str(path.resolve()), 'sha256': sha256(path)} for parameter, path in sample_masks.items()}, 'formal_endpoint_statement': 'This mask is used only to make the immediate raw NeSVoR volume export well-defined. The formal central/psf128 matched endpoints remain separate fixed-L100 checkpoint queries.'}
    commands=build_independent_nesvor_commands(exports, destination/'nesvor', nesvor_executable=nesvor_executable, sample_masks=sample_masks, device=device)
    (destination/'commands.sh').write_text('\n'.join(commands)+'\n')
    observations = ({stack: standardized_observations[stack] for stack in active_stacks} if standardized_mode else {stack: sha256(prepared/stack/'observations.npz') for stack in active_stacks})
    result_manifest = {'method':'fit_first','subject_id':subject_id,'observations':observations,'training_stacks':active_stacks,'dictionary':'frozen current HHZ/TRAD grid; normalized cosine; no interpolation/refinement','native_map_reuse':native_reuse,'normalization':{'nesvor_output_intensity_rescaling':'disabled via --output-intensity-mean 0','t1_input_normalization':'divide by 2500 ms','t1_output_restoration':'multiply by 2500 ms exactly once','t2_input_normalization':'divide by 200 ms','t2_output_restoration':'multiply by 200 ms exactly once'},'nesvor_version':'nesvor_v5_128_w1','historical_nesvor_executable':str(nesvor_executable),'nesvor':'independent T1/T2 stack registration and transformation optimization; no pose copying; output models saved per parameter for future resampling','commands_sha256':sha256(destination/'commands.sh')}
    if sparse_profile is not None:
        result_manifest.update({'profile_id': sparse_profile['profile_id'], 'case_id': sparse_profile['case_id'], 'fold_id': sparse_profile.get('fold_id'), 'fold_count': sparse_profile.get('fold_count'), 'sparse_manifest': {'path': sparse_profile['_manifest_path'], 'sha256': sparse_profile['_manifest_sha256'], 'generator_git_sha': sparse_profile['git_sha']}, 'endpoint_contract': {'central': 'central/no map-domain PSF post-hoc endpoint', 'psf128': 'PSF128 downstream endpoint uses the same subject/case identity'}})
    if raw_output_sampling_mask is not None:
        result_manifest['raw_output_sampling_mask'] = raw_output_sampling_mask
    if standardized_mode:
        result_manifest.update({
            'standardized_native_dictionary_reuse': native_reuse,
            'selected_standardized_native_maps': {'path': str((native_maps / 'manifest.json').resolve()), 'sha256': sha256(native_maps / 'manifest.json')},
            'selected_group_ids': {stack: record.get('standardized_group_ids', []) for stack, record in selected_standardized_records.items()},
            'formal_scmr_one_stage_candidate': sparse_profile is not None,
        })
    write_immutable_json(destination/'input_manifest.json', result_manifest)
    return destination, commands


def finalize_scmr_one_stage_fit_first(root: str | Path) -> Path:
    """Publish the formal one-stage SCMR route only after both NeSVoR jobs finish."""
    destination = Path(root)
    input_manifest = json.loads((destination / 'input_manifest.json').read_text(encoding='utf-8'))
    if input_manifest.get('formal_scmr_one_stage_candidate') is not True:
        raise ValueError('formal SCMR finalization requires standardized sparse one-stage input provenance')
    commands = destination / 'commands.sh'
    if not commands.is_file():
        raise FileNotFoundError(commands)
    command_text = commands.read_text(encoding='utf-8')
    if '--registration stack' not in command_text or '--no-transformation-optimization' in command_text:
        raise ValueError('one-stage SCMR commands do not prove stack registration with transformation optimization')
    parameters = {}
    for parameter in ('t1', 't2'):
        target = destination / 'nesvor' / parameter
        model, result, slices = target / 'model.pt', target / 'result.json', target / 'slices'
        if not model.is_file() or not result.is_file() or not slices.is_dir() or not any(slices.glob('[0-9]*.nii.gz')):
            raise FileNotFoundError(f'{parameter}: one-stage NeSVoR model/result/final slices are incomplete')
        parameters[parameter] = {'model': {'path': str(model.resolve()), 'sha256': sha256(model)}, 'result': {'path': str(result.resolve()), 'sha256': sha256(result)}, 'final_slices': str(slices.resolve())}
    sparse = input_manifest.get('sparse_manifest')
    selected = input_manifest.get('selected_standardized_native_maps')
    if not isinstance(sparse, dict) or not isinstance(selected, dict):
        raise ValueError('formal SCMR route sparse/selected-map provenance is absent')
    manifest = {
        'schema': 'phase4r_scmr_one_stage_fit_first/v1',
        'subject': input_manifest['subject_id'], 'case_id': input_manifest['case_id'], 'profile_id': input_manifest['profile_id'],
        'fold_id': input_manifest.get('fold_id'), 'fold_count': input_manifest.get('fold_count'),
        'sparse_manifest': sparse, 'standardized_native_dictionary_reuse': input_manifest['standardized_native_dictionary_reuse'],
        'selected_standardized_native_maps': selected, 'selected_group_ids': input_manifest['selected_group_ids'],
        'active_stacks': input_manifest['training_stacks'], 'no_dictionary_refit': True,
        'normalization': input_manifest['normalization'], 'commands': {'path': str(commands.resolve()), 'sha256': sha256(commands)},
        'exact_nesvor_commands': [line for line in command_text.splitlines() if line],
        'nesvor_version': 'nesvor_v5_128_w1', 'historical_reference': 'SCMR one-stage independent NeSVoR baseline',
        'independent_t1_t2': True, 'transformation_optimization_enabled': True,
        'no_slice_scale': True, 'parameters': parameters,
        'code_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    }
    return write_immutable_json(destination / 'manifest.json', manifest)


def build_baseline_command(args: argparse.Namespace) -> str:
    """Return the future Fit-first command without creating native-map outputs."""
    values = [sys.executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.baseline", "--prepared-root", args.prepared_root, "--subject-id", args.subject_id, "--output", args.output, "--nesvor-executable", args.nesvor_executable]
    if args.reuse_native_maps_root:
        values.extend(("--reuse-native-maps-root", args.reuse_native_maps_root))
    if args.sparse_manifest:
        values.extend(("--sparse-manifest", args.sparse_manifest, "--expected-generator-git-sha", args.expected_generator_git_sha, "--expected-manifest-sha256", args.expected_manifest_sha256))
    if args.standardized_native_maps_root:
        values.extend(("--standardized-native-maps-root", args.standardized_native_maps_root, "--standardized-reuse-manifest", args.standardized_reuse_manifest))
    values.extend(("--device", args.device))
    values.append("--execute")
    return " ".join(shlex.quote(str(value)) for value in values)

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepared-root',required=True); parser.add_argument('--subject-id',required=True); parser.add_argument('--output',required=True); parser.add_argument('--nesvor-executable',default=HISTORICAL_NESVOR_EXECUTABLE_DEFAULT); parser.add_argument('--reuse-native-maps-root'); parser.add_argument('--sparse-manifest'); parser.add_argument('--standardized-native-maps-root'); parser.add_argument('--standardized-reuse-manifest'); parser.add_argument('--expected-generator-git-sha'); parser.add_argument('--expected-manifest-sha256'); parser.add_argument('--device', default='0'); parser.add_argument('--execute',action='store_true'); args=parser.parse_args()
    if bool(args.expected_generator_git_sha) != bool(args.expected_manifest_sha256):
        parser.error("sparse Fit-first requires both generator and manifest SHA256 pins")
    if args.sparse_manifest and not (args.expected_generator_git_sha and args.expected_manifest_sha256):
        parser.error("sparse Fit-first requires --expected-generator-git-sha and --expected-manifest-sha256")
    if not args.execute:
        if args.sparse_manifest:
            validate_sparse_profile_manifest(args.sparse_manifest, subject=args.subject_id, strict=True, expected_generator_git_sha=args.expected_generator_git_sha, expected_manifest_sha256=args.expected_manifest_sha256)
        print(build_baseline_command(args)); return
    reuse = {'reuse_native_maps_root': args.reuse_native_maps_root, 'sparse_manifest': args.sparse_manifest, 'standardized_native_maps_root': args.standardized_native_maps_root, 'standardized_reuse_manifest': args.standardized_reuse_manifest} if args.reuse_native_maps_root or args.sparse_manifest or args.standardized_native_maps_root or args.standardized_reuse_manifest else {}
    pins = {"expected_generator_git_sha": args.expected_generator_git_sha, "expected_manifest_sha256": args.expected_manifest_sha256} if args.expected_generator_git_sha else {}
    root, commands=build_baseline_run(args.prepared_root,args.subject_id,args.output,args.nesvor_executable,device=args.device,**pins,**reuse)
    for parameter in ('t1','t2'): (root/'nesvor'/parameter).mkdir(parents=True, exist_ok=False)
    for command in commands: subprocess.run(command, shell=True, check=True)
    if (root / 'input_manifest.json').is_file() and json.loads((root / 'input_manifest.json').read_text(encoding='utf-8')).get('formal_scmr_one_stage_candidate') is True:
        finalize_scmr_one_stage_fit_first(root)
    print(root)
if __name__ == '__main__': main()
