"""Create the immutable Phase-2 clean input contract; never reconstruct."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
from .contracts import APPROVED_DECODER_SHA256, BASELINE_POSE_POLICY, ELIGIBLE_REAL_SUBJECTS, PROPOSED_CONTRACT, assert_phase1_ready, observation_hashes, sha256, write_immutable_json

def prepare(phase1_root: str|Path, output: str|Path, b6_model: str|Path, decoder_manifest: str|Path) -> Path:
    root=Path(phase1_root); gate=root/'phase1_final_gate_matlab_parity.json'; assert_phase1_ready(json.loads(gate.read_text()))
    decoder=json.loads(Path(decoder_manifest).read_text())
    if decoder.get('sha256') != APPROVED_DECODER_SHA256: raise ValueError('approved decoder manifest SHA256 differs from Phase-2 contract')
    manifest={"phase":"2_clean_full_preparation","code_commit":subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),"synthetic_root":str(root.resolve()),"phase1_final_gate":{"path":str(gate.resolve()),"sha256":sha256(gate)},"observation_sha256":observation_hashes(root),"gt_sha256":{"gt_maps.npz":sha256(root/'gt_maps.npz'),"mesh_primary_myo.npz":sha256(root/'mesh_primary_myo.npz')},"timing_sha256":sha256(root/'timing_selection.json'),"geometry_sha256":sha256(root/'geometry_templates.json'),"proposed":PROPOSED_CONTRACT,"baseline":{"dictionary_grid":"frozen current HHZ/TRAD grid; normalized cosine matching, no interpolation/refinement",**BASELINE_POSE_POLICY},"approved_decoder":{"sha256":APPROVED_DECODER_SHA256,"manifest":str(Path(decoder_manifest).resolve())},"b6_source_model":{"path":str(Path(b6_model).resolve()),"sha256":sha256(b6_model)},"real_paired_subjects":list(ELIGIBLE_REAL_SUBJECTS),"excluded":{"DYL":"VPS=85 incompatible with approved VPS=87 decoder"}}
    return write_immutable_json(output,manifest)

def main()->None:
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--phase1-root',required=True);p.add_argument('--output',required=True);p.add_argument('--b6-model',required=True);p.add_argument('--decoder-manifest',required=True);a=p.parse_args();print(prepare(**vars(a)))
if __name__=='__main__': main()
