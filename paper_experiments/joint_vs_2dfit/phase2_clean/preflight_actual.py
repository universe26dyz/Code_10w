"""Server-side actual-file Phase-2 integrity gate; it never creates a run directory."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .contracts import sha256, APPROVED_DECODER_SHA256

def verify(manifest: str|Path, data_root: str|Path, synthetic_root: str|Path, decoder: str|Path, b6_model: str|Path, reference_manifest: str|Path) -> dict:
 m=json.loads(Path(manifest).read_text()); data=Path(data_root); synthetic=Path(synthetic_root)
 actual={'synthetic':{},'real':{}}
 for stack,expected in m['observation_sha256'].items():
  p=synthetic/stack/'observations.npz'; actual['synthetic'][stack]={'path':str(p),'sha256':sha256(p)}
  if actual['synthetic'][stack]['sha256']!=expected: raise ValueError(f'synthetic {stack} SHA mismatch')
 for sid, stacks in m['real_observation_sha256'].items():
  actual['real'][sid]={}
  for stack,record in stacks.items():
   p=data/sid/stack/'observations.npz'; actual['real'][sid][stack]={'path':str(p),'sha256':sha256(p)}
   if actual['real'][sid][stack]['sha256']!=record['sha256']: raise ValueError(f'{sid}/{stack} SHA mismatch')
 d,b=Path(decoder),Path(b6_model)
 if sha256(d)!=APPROVED_DECODER_SHA256: raise ValueError('approved decoder SHA mismatch')
 if sha256(b)!='15981fbe93d2c70e14755ea756a02856c9383481e7d92d239619268a628a29b5': raise ValueError('B6 source SHA mismatch')
 refs=json.loads(Path(reference_manifest).read_text())
 for sid,record in refs['subjects'].items():
  for key in ('native_reference','myocardium_bundle'):
   path=Path(record[key]['path'])
   if not path.is_file() or sha256(path)!=record[key]['sha256']: raise ValueError(f'{sid} {key} missing or SHA mismatch')
 return {'actual_observation_hash_gate':True,'references_ready':True,'approved_decoder_gate':True,'b6_config_source_gate':True,'actual_observations':actual,'approved_decoder':{'path':str(d),'sha256':sha256(d)},'b6_source_model':{'path':str(b),'sha256':sha256(b)}}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--reference-manifest',required=True);p.add_argument('--data-root',required=True);p.add_argument('--synthetic-root',required=True);p.add_argument('--decoder',required=True);p.add_argument('--b6-model',required=True);p.add_argument('--output',required=True);a=p.parse_args(); out=Path(a.output)
 if out.exists(): raise FileExistsError(f'refusing overwrite: {out}')
 value=verify(a.manifest,a.data_root,a.synthetic_root,a.decoder,a.b6_model,a.reference_manifest);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(value,indent=2));print(out)
if __name__=='__main__':main()
