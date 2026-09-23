"""Validate an RR surrogate on explicit prepared VPS=87 timing sources."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset
from .mlp_model import MdmSignalMLP
from .synthetic_dataset import protocol_from_record
from .test_mlp import fidelity_metrics
from .trad_teacher.trad_signal_simulator import TradSignalSimulator

def _sources(root: Path, subjects: list[str]):
    rows=[]
    for subject in subjects:
        for stack in ("sax","2ch","4ch"):
            path=root/subject/stack/"observations.npz"
            with np.load(path,allow_pickle=False) as data:
                groups=np.asarray(data["group_idx"],dtype=np.int64); timing=np.asarray(data["timing9_ms"],dtype=np.float32); tr=np.asarray(data["tr_ms"]); vps=np.asarray(data["vps"])
            for group in np.unique(groups):
                ix=np.flatnonzero(groups==group)
                if int(vps[ix[0]]) != 87: raise ValueError(f"{subject}/{stack}/group={group} has VPS={int(vps[ix[0]])}, expected 87.")
                if not np.allclose(timing[ix],timing[ix[0]],rtol=0,atol=1e-6): raise ValueError("group timing is inconsistent")
                rows.append({"subject":subject,"stack":stack,"group_idx":int(group),"tr_ms":float(tr[ix[0]]),"vps":int(vps[ix[0]]),"timing9_ms":timing[ix[0]]})
    return rows

def validate_real_timings(checkpoint_path, prepared_root, subjects, rr_dataset_dir, protocol_path, output_path, device_name="cpu", samples_per_timing=16):
    output=Path(output_path)
    csv_output = output.with_name("real_timing_validation_per_source.csv")
    if output.exists() or csv_output.exists():
        raise FileExistsError(f"Real timing validation outputs must be new: {output}, {csv_output}")
    device=torch.device(device_name); rows=_sources(Path(prepared_root),list(subjects)); meta=json.loads((Path(rr_dataset_dir)/"dataset_metadata.json").read_text())
    if meta.get("schema") != "mlp_rr_synthetic/v1" or meta.get("split_mode") != "rhythm":
        raise ValueError("Real timing validation requires the active mlp_rr_synthetic/v1 rhythm-disjoint dataset.")
    protocol=protocol_from_record(meta["protocol"],protocol_path); checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=False)
    if checkpoint.get("dataset_schema") != "mlp_rr_synthetic/v1" or checkpoint.get("dataset_split_mode") != "rhythm":
        raise ValueError("Real timing validation requires an active RR synthetic checkpoint, not a legacy subject/timing-pool checkpoint.")
    model=MdmSignalMLP().to(device); model.load_state_dict(checkpoint["state_dict"]); model.eval(); rng=np.random.default_rng(20260911); reports=[]
    for row in rows:
        n=int(samples_per_timing); t2=rng.uniform(5,200,n).astype("f4"); t1=rng.uniform(np.maximum(20,t2+1e-6),2500,n).astype("f4"); b1=rng.uniform(.1,1.2,n).astype("f4"); timing=np.repeat(row["timing9_ms"][None],n,0)
        x=torch.from_numpy(np.column_stack((t1/1000,t2/1000,b1,timing/1000)).astype("f4")); target=TradSignalSimulator()(torch.from_numpy(t1),torch.from_numpy(t2),torch.from_numpy(b1),torch.from_numpy(timing),protocol,normalize=True)
        metric=fidelity_metrics(model, TensorDataset(x, target), n, n, protocol=protocol); reports.append({k:v for k,v in row.items() if k!="timing9_ms"}|{"timing9_ms":row["timing9_ms"].tolist(),**metric})
    real=np.stack([r["timing9_ms"] for r in rows]); lo=np.asarray(meta["train_timing9_min_ms"]); hi=np.asarray(meta["train_timing9_max_ms"]); outside=(real<lo)|(real>hi)
    coverage_status = "PASS" if not outside.any() else "OUTSIDE_TRAINING_DOMAIN"
    result={"schema":"rr_real_vps87_validation/v1","sources":len(reports),"real_timing_min_ms":real.min(0).tolist(),"real_timing_max_ms":real.max(0).tolist(),"synthetic_train_timing_min_ms":lo.tolist(),"synthetic_train_timing_max_ms":hi.tolist(),"outside_per_dimension":outside.sum(0).astype(int).tolist(),"outside_fraction":float(outside.any(1).mean()),"training_domain_coverage_status":coverage_status,"approval_recommendation":"eligible_for_human_review" if coverage_status == "PASS" else "do_not_approve","per_source":reports,"validation_status":"awaiting_manual_review"}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2));
    with csv_output.open("w",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=["subject","stack","group_idx","tr_ms","vps","overall_rmse","mae","max_abs_error"]); writer.writeheader(); writer.writerows([{k:r[k] for k in ["subject","stack","group_idx","tr_ms","vps","overall_rmse","mae","max_abs_error"]} for r in reports])
    return result

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint",required=True);p.add_argument("--prepared-root",required=True);p.add_argument("--subjects",nargs="+",required=True);p.add_argument("--rr-dataset-dir",required=True);p.add_argument("--protocol",required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cpu");p.add_argument("--samples-per-timing",type=int,default=16);a=p.parse_args();print(json.dumps(validate_real_timings(a.checkpoint,a.prepared_root,a.subjects,a.rr_dataset_dir,a.protocol,a.output,a.device,a.samples_per_timing),indent=2))
if __name__=="__main__": main()
