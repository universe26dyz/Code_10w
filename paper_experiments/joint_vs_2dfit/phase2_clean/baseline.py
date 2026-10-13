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
        exports.append(export_nesvor_native_inputs(maps, destination/'nesvor_inputs'/stack, {'t1':2500.0,'t2':200.0}))
    commands=[]
    for parameter in ('t1','t2'):
        run_dir=destination/'nesvor'/parameter
        inputs=' '.join(str(item/parameter/'stack.nii.gz') for item in exports)
        scale = 2500 if parameter == 't1' else 200
        commands.append(f'nesvor reconstruct --input-stacks {inputs} --thicknesses 8 8 8 --registration stack --scanner-space --metric none --weight-transformation 1 --weight-deform 0.1 --weight-image 1 --weight-pixel-var-smooth 0 --output-volume {run_dir}/volume_normalized.nii.gz --output-json {run_dir}/result.json --output-log {run_dir}/nesvor.log --output-slices {run_dir}/slices --output-resolution 1.0 --output-psf-factor 1 --no-slice-scale --single-precision --log2-hashmap-size 19 --batch-size 2000 --n-iter 10000 --n-samples 256 --inference-samples 128 --inference-batch-size 16000 --pixel-variance-weight 1 --coarsest-resolution 16 --finest-resolution 0.5 --level-scale 1.3819 --n-features-per-level 2 --depth 1 --width 64 --n-features-z 15 --n-features-slice 16 --weight-bias 100 --image-regularization edge --delta 0.2 --learning-rate 0.005 --gamma 0.33 --milestones 0.5 0.75 0.9 --seed 20260823 --device 0; python -m paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_denormalize --input {run_dir}/volume_normalized.nii.gz --scale {scale} --output {run_dir}/volume_ms.nii.gz')
    (destination/'commands.sh').write_text('\n'.join(commands)+'\n')
    write_immutable_json(destination/'input_manifest.json', {'method':'fit_first','subject_id':subject_id,'observations':observation_hashes(prepared),'dictionary':'frozen current HHZ/TRAD grid; normalized cosine; no interpolation/refinement','normalization':{'t1_input_divisor_ms':2500,'t2_input_divisor_ms':200,'output':'multiply normalized output to ms before evaluation'},'nesvor_version':'nesvor_v5_128_w1','nesvor':'independent T1/T2 stack registration and transformation optimization; no pose copying','commands_sha256':sha256(destination/'commands.sh')})
    return destination, commands

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepared-root',required=True); parser.add_argument('--subject-id',required=True); parser.add_argument('--output',required=True); parser.add_argument('--execute',action='store_true'); args=parser.parse_args()
    root, commands=build_baseline_run(args.prepared_root,args.subject_id,args.output)
    if args.execute:
        for command in commands: subprocess.run(command, shell=True, check=True)
    print(root)
if __name__ == '__main__': main()
