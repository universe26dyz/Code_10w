"""Native current-protocol dictionary maps for the independent-NeSVoR baseline."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .contracts import BASELINE_POSE_POLICY, sha256, write_immutable_json
from .dictionary import frozen_dictionary, match_fingerprints

def fit_native(observations: str|Path, output: str|Path) -> Path:
    source,output=Path(observations),Path(output)
    if output.exists(): raise FileExistsError(f'refusing overwrite: {output}')
    data=np.load(source); groups=np.unique(data['group_idx']); h,w=data['images'].shape[-2:]
    t1=np.full((len(groups),h,w),np.nan,np.float32);t2=t1.copy();b1=t1.copy(); cache={}
    for out_group,group in enumerate(groups):
        rows=np.flatnonzero(data['group_idx']==group)
        if rows.size!=10 or not np.array_equal(np.sort(data['weight_idx'][rows]),np.arange(10)): raise ValueError('each group must include weights 0..9')
        ordered=rows[np.argsort(data['weight_idx'][rows])]; timing=tuple(np.asarray(data['timing9_ms'][ordered[0]],float)); tr=float(data['tr_ms'][ordered[0]]); vps=int(data['vps'][ordered[0]])
        if any(not np.array_equal(data['timing9_ms'][x],timing) or float(data['tr_ms'][x])!=tr or int(data['vps'][x])!=vps for x in ordered): raise ValueError('group timing/TR/VPS must be shared')
        key=(timing,tr,vps)
        if key not in cache: cache[key]=frozen_dictionary(np.asarray(timing),tr,vps)
        valid=np.all(data['masks'][ordered].astype(bool),axis=0)&np.isfinite(data['images'][ordered]).all(axis=0)
        idx=match_fingerprints(cache[key]['fingerprints'],np.moveaxis(data['images'][ordered,:, :],0,-1)[valid])
        for target,name in ((t1,'t1_ms'),(t2,'t2_ms'),(b1,'b1')): target[out_group][valid]=cache[key][name][idx]
    output.mkdir(parents=True); np.savez_compressed(output/'native_maps.npz',t1_ms=t1,t2_ms=t2,b1=b1,group_idx=groups)
    write_immutable_json(output/'manifest.json',{"method":"current-protocol HHZ/TRAD dictionary matching; normalized 10D inner product; no interpolation/refinement","source_observations":{"path":str(source.resolve()),"sha256":sha256(source)},"dictionary_cache_entries":len(cache),"baseline_pose_policy":BASELINE_POSE_POLICY,"next_formal_step":"independent T1 and T2 NeSVoR stack registration/transformation optimization"})
    return output
def main()->None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--observations',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(fit_native(**vars(a)))
if __name__=='__main__':main()
