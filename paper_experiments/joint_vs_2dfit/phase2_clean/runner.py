"""Formal-run manifest guard; server scripts call it before reconstruction."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .contracts import ELIGIBLE_REAL_SUBJECTS, compare_observation_hashes, validate_proposed_subject
def preflight(method:str,dataset:str,input_manifest:str|Path,method_manifest:str|Path,subject_id:str|None=None)->dict:
 if method not in ('proposed','fit_first') or dataset not in ('synthetic_clean','real_full'): raise ValueError('method/dataset outside Phase-2 contract')
 shared=json.loads(Path(input_manifest).read_text()); method_data=json.loads(Path(method_manifest).read_text()); compare_observation_hashes(shared['observation_sha256'],method_data['observation_sha256'])
 if dataset=='real_full':
  if subject_id not in ELIGIBLE_REAL_SUBJECTS: raise ValueError('subject not in Phase-2 paired real cohort')
  validate_proposed_subject(subject_id,2.61,87) if method=='proposed' else None
 return {'status':'preflight_pass','method':method,'dataset':dataset,'subject_id':subject_id}
def main()->None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',required=True);p.add_argument('--dataset',required=True);p.add_argument('--input-manifest',required=True);p.add_argument('--method-manifest',required=True);p.add_argument('--subject-id');a=p.parse_args();print(json.dumps(preflight(**vars(a)),indent=2))
if __name__=='__main__':main()
