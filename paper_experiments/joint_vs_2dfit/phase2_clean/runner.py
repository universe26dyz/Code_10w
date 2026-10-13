"""Formal-run manifest guard; server scripts call it before reconstruction."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .contracts import ELIGIBLE_REAL_SUBJECTS, compare_observation_hashes, compare_real_observation_identities, validate_proposed_subject
def _load(value: str | Path | dict) -> dict:
 return value if isinstance(value, dict) else json.loads(Path(value).read_text())
def preflight(method:str,dataset:str,input_manifest:str|Path|dict,method_manifest:str|Path|dict,subject_id:str|None=None)->dict:
 if method not in ('proposed','fit_first') or dataset not in ('synthetic_clean','real_full'): raise ValueError('method/dataset outside Phase-2 contract')
 shared=_load(input_manifest); method_data=_load(method_manifest)
 if dataset == 'synthetic_clean': compare_observation_hashes(shared['observation_sha256'],method_data['observation_sha256'])
 if dataset=='real_full':
  if subject_id not in ELIGIBLE_REAL_SUBJECTS: raise ValueError('subject not in Phase-2 paired real cohort')
  compare_real_observation_identities(shared['real_observation_sha256'], method_data['real_observation_sha256'], subject_id)
  validate_proposed_subject(subject_id,2.61,87) if method=='proposed' else None
 return {'status':'preflight_pass','method':method,'dataset':dataset,'subject_id':subject_id}
def main()->None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',required=True);p.add_argument('--dataset',required=True);p.add_argument('--input-manifest',required=True);p.add_argument('--method-manifest',required=True);p.add_argument('--subject-id');a=p.parse_args();print(json.dumps(preflight(**vars(a)),indent=2))
if __name__=='__main__':main()
