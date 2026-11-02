"""Create final 2x2 SCMR AHA bullseye only after AHA validation and approval."""
from __future__ import annotations
import argparse,json
import matplotlib.pyplot as plt
from evaluation.utils.common import load_yaml,output_root,read_csv
from visualization.plotting import bullseye,parameter_limits,save_figure
def run(cfg,overwrite=False):
 root=output_root(cfg);validation=root/"manifests/aha16_validation.json";selection=load_yaml(cfg["aha"]["selection_file"]).get("bullseye",{})
 if not validation.exists() or json.loads(validation.read_text()).get("status")!="PASS" or not selection.get("qc_approved",False):
  print("SKIPPED: AHA QC not passed/approved; final Figure 3 was not generated");return None
 rows=read_csv(root/"aha16/aha16_stats.csv");fig,axes=plt.subplots(2,2,figsize=(8,8))
 for col,p in enumerate(("T1","T2")):
  sub=[r for r in rows if r["parameter"]==p];lim=parameter_limits(cfg,p);orig={int(r["segment"]):float(r["original_mean"]) for r in sub};syn={int(r["segment"]):float(r["reprojection_mean"]) for r in sub}
  m=bullseye(axes[0,col],orig,*lim,title=f"{p} Original 2D");bullseye(axes[1,col],syn,*lim,title=f"{p} 3D PSF reprojection");fig.colorbar(m,ax=axes[:,col],fraction=.025,label="ms")
 fig.suptitle(f"{cfg['subject_id']} AHA16 myocardial regional means");path=root/"figures/scmr/Figure3_bullseye_T1_T2.png"
 if not path.exists() or overwrite:save_figure(fig,path,int(cfg.get("figures",{}).get("dpi",300)),cfg.get("figures",{}).get("pdf",True))
 return path
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True);p.add_argument("--overwrite",action="store_true");a=p.parse_args();print(run(load_yaml(a.config),a.overwrite))
if __name__=="__main__":main()
