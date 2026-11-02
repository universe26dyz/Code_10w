"""Phase-2 adapter for the audited Q002/B6 production CLI; it never starts a warm start."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
from .contracts import APPROVED_DECODER_SHA256, PROPOSED_CONTRACT, sha256

def build_proposed_command(prepared_root: str|Path, decoder: str|Path, b6_model: str|Path, output: str|Path, subject_id: str = 'CYJ', device: str = 'cuda:0', *, protocol: str|Path = 'trad/configs/protocol_hhz_v1.yaml', sparse_manifest: str|Path|None = None, expected_sparse_generator_git_sha: str|None = None, expected_sparse_manifest_sha256: str|None = None, expected_b6_model_sha256: str|None = None, expected_decoder_sha256: str|None = None, expected_protocol_sha256: str|None = None, endpoint_policy: str = 'legacy') -> str:
    decoder = Path(decoder)
    expected_decoder_sha256 = expected_decoder_sha256 or APPROVED_DECODER_SHA256
    if sha256(decoder) != expected_decoder_sha256: raise ValueError('decoder path does not match expected decoder SHA256')
    args = [sys.executable,'-m','quality_experiments.Q002_B6_roi_same_anchor_mse.run_q002_reconstruction','--cropped-prepared-root',str(Path(prepared_root)),'--subject-id',subject_id,'--b6-model',str(Path(b6_model)),'--signal-simulator',str(decoder),'--expected-decoder-sha256',expected_decoder_sha256,'--protocol',str(Path(protocol)),'--output',str(Path(output)),'--device',device,'--experiment-id','Q002S640_B6_roi_same_anchor_mse','--anchor-batch-size','640','--endpoint-policy',endpoint_policy]
    if sparse_manifest is not None:
        args += ['--sparse-manifest', str(Path(sparse_manifest))]
    for flag, value in (('--expected-sparse-generator-git-sha', expected_sparse_generator_git_sha), ('--expected-sparse-manifest-sha256', expected_sparse_manifest_sha256), ('--expected-b6-model-sha256', expected_b6_model_sha256), ('--expected-protocol-sha256', expected_protocol_sha256)):
        if value is not None: args += [flag, value]
    return ' '.join(__import__('shlex').quote(value) for value in args)

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepared-root',required=True); parser.add_argument('--decoder',required=True); parser.add_argument('--b6-model',required=True); parser.add_argument('--output',required=True); parser.add_argument('--subject-id',default='CYJ'); parser.add_argument('--device',default='cuda:0'); parser.add_argument('--protocol',default='trad/configs/protocol_hhz_v1.yaml'); parser.add_argument('--sparse-manifest'); parser.add_argument('--expected-sparse-generator-git-sha'); parser.add_argument('--expected-sparse-manifest-sha256'); parser.add_argument('--expected-b6-model-sha256'); parser.add_argument('--expected-decoder-sha256'); parser.add_argument('--expected-protocol-sha256'); parser.add_argument('--endpoint-policy',choices=('legacy','central_only'),default='legacy'); parser.add_argument('--execute',action='store_true'); args=parser.parse_args()
    command=build_proposed_command(args.prepared_root,args.decoder,args.b6_model,args.output,args.subject_id,args.device,protocol=args.protocol,sparse_manifest=args.sparse_manifest,expected_sparse_generator_git_sha=args.expected_sparse_generator_git_sha,expected_sparse_manifest_sha256=args.expected_sparse_manifest_sha256,expected_b6_model_sha256=args.expected_b6_model_sha256,expected_decoder_sha256=args.expected_decoder_sha256,expected_protocol_sha256=args.expected_protocol_sha256,endpoint_policy=args.endpoint_policy)
    if not args.execute: print(command); return
    subprocess.run(command, shell=True, check=True)
if __name__ == '__main__': main()
