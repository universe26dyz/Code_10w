"""Synthetic mesh-GT and real native-reference agreement metrics."""
from __future__ import annotations
import numpy as np

def _metrics(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict:
    r,p=np.asarray(reference,float)[mask],np.asarray(prediction,float)[mask]
    if not len(r): raise ValueError("empty evaluation support")
    diff=p-r; rmse=float(np.sqrt(np.mean(diff**2))); rng=float(np.max(r)-np.min(r))
    pearson=float(np.corrcoef(r,p)[0,1]) if len(r)>1 and np.std(r)>0 and np.std(p)>0 else float("nan")
    ncc=float(np.dot(r-r.mean(),p-p.mean())/(np.linalg.norm(r-r.mean())*np.linalg.norm(p-p.mean()))) if np.std(r)>0 and np.std(p)>0 else float("nan")
    return {"bias":float(diff.mean()),"mae":float(np.abs(diff).mean()),"rmse":rmse,"nrmse":rmse/rng if rng else float("nan"),"valid_mape":float(np.mean(np.abs(diff[r!=0]/r[r!=0]))*100) if np.any(r!=0) else float("nan"),"pearson_r":pearson,"ncc":ncc,"ssim":None,"support_voxels":int(len(r))}

def synthetic_metrics(reference: np.ndarray, prediction: np.ndarray, myocardium_full: np.ndarray, valid: np.ndarray) -> dict:
    support=np.asarray(myocardium_full,bool)&np.asarray(valid,bool)&np.isfinite(reference)&np.isfinite(prediction)
    report=_metrics(reference,prediction,support); report.update({"primary_region":"full_mesh_myocardium","coverage_fraction":float(support.sum()/np.asarray(myocardium_full,bool).sum()),"out_of_grid_voxels":int((np.asarray(myocardium_full,bool)&~np.asarray(valid,bool)).sum()),"world_alignment":"raw known world geometry primary; common SE(3) diagnostic only"}); return report

def real_native_agreement(reference: np.ndarray, prediction: np.ndarray, myocardium_mask: np.ndarray) -> dict:
    report=_metrics(reference,prediction,np.asarray(myocardium_mask,bool)&np.isfinite(reference)&np.isfinite(prediction)); report["endpoint"]="native-reference agreement/consistency, not accuracy versus ground truth"; return report
