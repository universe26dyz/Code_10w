"""Phase-2 adapter for the audited Q002/B6 production CLI; it never starts a warm start."""
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
from .contracts import APPROVED_DECODER_SHA256, PROPOSED_CONTRACT, sha256

def build_proposed_command(prepared_root: str|Path, decoder: str|Path, b6_model: str|Path, output: str|Path, subject_id: str = 'CYJ', device: str = 'cuda:0') -> str:
    decoder = Path(decoder)
    if sha256(decoder) != APPROVED_DECODER_SHA256: raise ValueError('decoder path does not have approved SHA256')
    args = ['python','-m','quality_experiments.Q002_B6_roi_same_anchor_mse.run_q002_reconstruction','--cropped-prepared-root',str(Path(prepared_root)),'--subject-id',subject_id,'--b6-model',str(Path(b6_model)),'--signal-simulator',str(decoder),'--output',str(Path(output)),'--device',device,'--experiment-id','Q002S640_B6_roi_same_anchor_mse','--anchor-batch-size','640']
    return ' '.join(__import__('shlex').quote(value) for value in args)

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepared-root',required=True); parser.add_argument('--decoder',required=True); parser.add_argument('--b6-model',required=True); parser.add_argument('--output',required=True); parser.add_argument('--subject-id',default='CYJ'); parser.add_argument('--device',default='cuda:0'); parser.add_argument('--execute',action='store_true'); args=parser.parse_args()
    command=build_proposed_command(args.prepared_root,args.decoder,args.b6_model,args.output,args.subject_id,args.device)
    if not args.execute: print(command); return
    subprocess.run(command, shell=True, check=True)
if __name__ == '__main__': main()
