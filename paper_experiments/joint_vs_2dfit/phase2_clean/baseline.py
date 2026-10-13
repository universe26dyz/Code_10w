"""Formal fit-first driver: dictionary maps then independent T1/T2 NeSVoR executions."""
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
from .contracts import HISTORICAL_NESVOR_EXECUTABLE_DEFAULT, observation_hashes, sha256, write_immutable_json
from .fit_first import fit_native, export_nesvor_native_inputs, build_independent_nesvor_commands

def build_baseline_run(prepared_root: str|Path, subject_id: str, output: str|Path, nesvor_executable: str|Path = HISTORICAL_NESVOR_EXECUTABLE_DEFAULT) -> tuple[Path, list[str]]:
    base, destination = Path(prepared_root), Path(output)
    prepared = base if all((base/stack/'observations.npz').is_file() for stack in ('sax','2ch','4ch')) else base/subject_id
    if destination.exists(): raise FileExistsError(f'refusing overwrite: {destination}')
    destination.mkdir(parents=True)
    exports=[]
    for stack in ('sax','2ch','4ch'):
        maps=fit_native(prepared/stack/'observations.npz', destination/'native_maps'/stack)
        exports.append(export_nesvor_native_inputs(maps, destination/'nesvor_inputs'/stack, {'t1':2500.0,'t2':200.0}))
    commands=build_independent_nesvor_commands(exports, destination/'nesvor', nesvor_executable=nesvor_executable)
    (destination/'commands.sh').write_text('\n'.join(commands)+'\n')
    write_immutable_json(destination/'input_manifest.json', {'method':'fit_first','subject_id':subject_id,'observations':observation_hashes(prepared),'dictionary':'frozen current HHZ/TRAD grid; normalized cosine; no interpolation/refinement','normalization':{'t1_input_divisor_ms':2500,'t2_input_divisor_ms':200,'output':'multiply normalized output to ms before evaluation'},'nesvor_version':'nesvor_v5_128_w1','historical_nesvor_executable':str(nesvor_executable),'nesvor':'independent T1/T2 stack registration and transformation optimization; no pose copying','commands_sha256':sha256(destination/'commands.sh')})
    return destination, commands

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepared-root',required=True); parser.add_argument('--subject-id',required=True); parser.add_argument('--output',required=True); parser.add_argument('--nesvor-executable',default=HISTORICAL_NESVOR_EXECUTABLE_DEFAULT); parser.add_argument('--execute',action='store_true'); args=parser.parse_args()
    root, commands=build_baseline_run(args.prepared_root,args.subject_id,args.output,args.nesvor_executable)
    if args.execute:
        for command in commands: subprocess.run(command, shell=True, check=True)
    print(root)
if __name__ == '__main__': main()
