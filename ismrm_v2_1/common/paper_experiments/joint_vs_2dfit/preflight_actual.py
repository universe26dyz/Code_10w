"""Server-side actual-file Phase-2 integrity gate; it never creates a run directory."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .contracts import sha256, APPROVED_DECODER_SHA256, B6_CONFIG_SOURCE_SHA256, CANONICAL_APPROVED_DECODER_DECLARED_PATH, CANONICAL_APPROVED_DECODER_RELATIVE_PATH
from ..xcat3d.finalize import assert_final_manifest_ready

CANONICAL_DECODER_RELATIVE = Path(CANONICAL_APPROVED_DECODER_RELATIVE_PATH)


def verify_approved_decoder(decoder: str|Path, audit_manifest: str|Path, data_root: str|Path) -> dict:
 d, audit, root = Path(decoder), Path(audit_manifest), Path(data_root).resolve()
 if not d.is_file(): raise FileNotFoundError(f'canonical approved decoder missing: {d}')
 payload=json.loads(audit.read_text())
 if payload.get('approved_for_paper') is not True: raise ValueError('approved decoder provenance is not approved_for_paper')
 if payload.get('approved_by_manual_review') is not True: raise ValueError('approved decoder provenance is not approved_by_manual_review')
 if payload.get('validation_status') != 'approved_by_manual_review': raise ValueError('approved decoder provenance validation status mismatch')
 if payload.get('sha256') != APPROVED_DECODER_SHA256: raise ValueError('approved decoder provenance SHA mismatch')
 if payload.get('resolved_from_provenance',{}).get('declared_decoder_path') != CANONICAL_APPROVED_DECODER_DECLARED_PATH: raise ValueError('approved decoder provenance declared canonical path mismatch')
 if d.resolve() != (root / CANONICAL_DECODER_RELATIVE).resolve(): raise ValueError(f'approved decoder is not the canonical DATA-root path: {d}')
 actual=sha256(d)
 if actual!=APPROVED_DECODER_SHA256: raise ValueError('approved decoder SHA mismatch')
 return {'path':str(d),'sha256':actual,'authority':'canonical_path_plus_sha256','audit_manifest':{'path':str(audit.resolve()),'sha256':sha256(audit)}}

def verify(manifest: str|Path, data_root: str|Path, synthetic_root: str|Path, decoder: str|Path, b6_model: str|Path, reference_manifest: str|Path, reference_root: str|Path, approved_decoder_audit: str|Path) -> dict:
 m=json.loads(Path(manifest).read_text()); data=Path(data_root); synthetic=Path(synthetic_root)
 source=assert_final_manifest_ready(synthetic)
 actual={'synthetic':{},'real':{}}
 for stack in ('sax','2ch','4ch'):
  p=synthetic/stack/'observations.npz'; actual['synthetic'][stack]={'path':str(p),'sha256':sha256(p)}
 for sid, stacks in m['real_observation_sha256'].items():
  actual['real'][sid]={}
  for stack,record in stacks.items():
   p=data/sid/stack/'observations.npz'; actual['real'][sid][stack]={'path':str(p),'sha256':sha256(p)}
   if actual['real'][sid][stack]['sha256']!=record['sha256']: raise ValueError(f'{sid}/{stack} SHA mismatch')
 d,b=Path(decoder),Path(b6_model)
 approved=verify_approved_decoder(d, approved_decoder_audit, reference_root)
 if sha256(b)!=B6_CONFIG_SOURCE_SHA256: raise ValueError('B6 source SHA mismatch')
 refs=json.loads(Path(reference_manifest).read_text())
 for sid,record in refs['subjects'].items():
  if sid not in actual['real']: raise ValueError(f'{sid} is not in actual prepared observations')
  if record.get('sax_observation_sha256') != actual['real'][sid]['sax']['sha256']:
   raise ValueError(f'{sid} reference source SAX SHA mismatch')
  for key in ('native_reference','myocardium_bundle'):
   path=Path(record[key].get('relative_path', record[key].get('path')))
   if not path.is_absolute(): path=Path(reference_root) / path
   if not path.is_file() or sha256(path)!=record[key]['sha256']: raise ValueError(f'{sid} {key} missing or SHA mismatch')
   if key == 'native_reference':
    native=json.loads(path.read_text())
    if native.get('current_protocol_source_observations',{}).get('sha256') != actual['real'][sid]['sax']['sha256']:
     raise ValueError(f'{sid} native-reference internal source SAX SHA mismatch')
 return {'actual_observation_hash_gate':True,'phase1_2_final_readiness_gate':True,'references_ready':True,'approved_decoder_gate':True,'b6_config_source_gate':True,'actual_observations':actual,'phase1_2_final_manifest':source,'approved_decoder':approved,'b6_source_model':{'path':str(b),'sha256':sha256(b)},'resolved_reference_root':str(Path(reference_root).resolve()),'reference_manifest':{'path':str(Path(reference_manifest).resolve()),'sha256':sha256(reference_manifest)},'reference_subjects':refs['subjects']}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--reference-manifest',required=True);p.add_argument('--reference-root',required=True);p.add_argument('--data-root',required=True);p.add_argument('--synthetic-root',required=True);p.add_argument('--decoder',required=True);p.add_argument('--approved-decoder-audit',required=True);p.add_argument('--b6-model',required=True);p.add_argument('--output',required=True);a=p.parse_args(); out=Path(a.output)
 if out.exists(): raise FileExistsError(f'refusing overwrite: {out}')
 value=verify(a.manifest,a.data_root,a.synthetic_root,a.decoder,a.b6_model,a.reference_manifest,a.reference_root,a.approved_decoder_audit);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(value,indent=2));print(out)
if __name__=='__main__':main()
