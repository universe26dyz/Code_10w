"""Write a ms-valued NIfTI from a normalized NeSVoR volume without changing geometry."""
from __future__ import annotations
import argparse
from pathlib import Path
def denormalize(input: str|Path, scale: float, output: str|Path) -> Path:
 import nibabel as nib
 source,target=Path(input),Path(output)
 if target.exists(): raise FileExistsError(f'refusing overwrite: {target}')
 image=nib.load(str(source)); nib.save(nib.Nifti1Image(image.get_fdata()*float(scale),image.affine,image.header),str(target)); return target
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--scale',required=True,type=float);p.add_argument('--output',required=True);a=p.parse_args();print(denormalize(**vars(a)))
if __name__=='__main__':main()
