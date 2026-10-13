"""Formal fit-first driver: dictionary maps then independent T1/T2 NeSVoR executions."""
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
from .contracts import observation_hashes, sha256, write_immutable_json
from .fit_first import fit_native, export_nesvor_native_inputs

def build_baseline_run(prepared_root: str|Path, subject_id: str, output: str|Path) -> tuple[Path, list[str]]:
    base, destination = Path(prepared_root), Path(output)
    prepared = base if all((base/stack/'observations.npz').is_file() for stack in ('sax','2ch','4ch')) else base/subject_id
    if destination.exists(): raise FileExistsError(f'refusing overwrite: {destination}')
    destination.mkdir(parents=True)
    exports=[]
    for stack in ('sax','2ch','4ch'):
        maps=fit_native(prepared/stack/'observations.npz', destination/'native_maps'/stack)
        exports.append(export_nesvor_native_inputs(maps, destination/'nesvor_inputs'/stack))
    commands=[]
    for parameter in ('t1','t2'):
        run_dir=destination/'nesvor'/parameter
        inputs=' '.join(str(item/parameter/'stack.nii.gz') for item in exports)
        commands.append(f'nesvor reconstruct --input-stacks {inputs} --registration stack --output-volume {run_dir}/volume.nii.gz --output-json {run_dir}/result.json --output-log {run_dir}/nesvor.log --output-slices {run_dir}/slices --output-resolution 1.0 --no-slice-scale --n-iter 10000 --n-samples 256 --inference-samples 128 --pixel-variance-weight 1.0 --device 0')
    (destination/'commands.sh').write_text('\n'.join(commands)+'\n')
    write_immutable_json(destination/'input_manifest.json', {'method':'fit_first','subject_id':subject_id,'observations':observation_hashes(prepared),'dictionary':'frozen current HHZ/TRAD grid; normalized cosine; no interpolation/refinement','nesvor':'independent T1/T2 stack registration and transformation optimization; no pose copying','commands_sha256':sha256(destination/'commands.sh')})
    return destination, commands

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepared-root',required=True); parser.add_argument('--subject-id',required=True); parser.add_argument('--output',required=True); parser.add_argument('--execute',action='store_true'); args=parser.parse_args()
    root, commands=build_baseline_run(args.prepared_root,args.subject_id,args.output)
    if args.execute:
        for command in commands: subprocess.run(command, shell=True, check=True)
    print(root)
if __name__ == '__main__': main()
