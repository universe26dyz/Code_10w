"""Run the complete evaluation pipeline for one subject in dependency order."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from evaluation.utils.common import dump_json,load_yaml,output_root,validate_config
from evaluation.geometry.build_slice_correspondence import run as correspondence
from evaluation.geometry.validate_registered_geometry import run as geometry
from evaluation.reprojection.generate_psf_matched_reprojections import run as reproject
from evaluation.reprojection.prepare_common_evaluation_mask import run as common_mask
from evaluation.reprojection.refine_common_support_to_central_component import run as central_mask
from evaluation.metrics.evaluate_psf_matched_reprojection import run as slice_metrics
from evaluation.metrics.summarize_metrics import run as summarize
from evaluation.roi.build_myocardium_masks import run as build_roi
from evaluation.roi.map_native_roi_to_final_pose import run as map_roi
from evaluation.roi.validate_roi_geometry import run as validate_roi
from evaluation.metrics.evaluate_myocardial_quantitative_fidelity import run as myocardial
from evaluation.metrics.compute_bland_altman import run as bland_altman
from evaluation.aha16.build_aha16_masks import run as build_aha
from evaluation.aha16.compute_aha16_stats import run as aha_stats
from evaluation.aha16.validate_aha16 import run as validate_aha
from visualization.qc.make_figure1_candidate_montages import run as candidates1
from visualization.qc.make_figure2_candidate_montages import run as candidates2
from visualization.individual.make_per_slice_reprojection_figure import run as individual
from visualization.individual.make_separate_mapping_images import run as separate_mapping
from visualization.individual.make_mapping_row_shared_bar import run as mapping_rows
from visualization.summary.plot_metric_summary import run as metric_plot
from visualization.qc.make_roi_final_pose_qc import run as roi_qc
from visualization.qc.make_aha16_qc import run as aha_qc
from visualization.summary.plot_bland_altman import run as ba_plot
from visualization.summary.plot_bullseye import run as bull_plot

def run(cfg, overwrite=False, quick=False):
    validate_config(cfg)
    steps = []

    def call(name, fn, *args):
        try:
            result = fn(cfg, overwrite, *args)
            steps.append({"step": name, "status": "COMPLETE", "result": str(result)})
        except FileExistsError as exc:
            steps.append({"step": name, "status": "EXISTS", "reason": str(exc)})
            return
        except Exception as exc:
            steps.append({"step": name, "status": "FAILED",
                          "reason": f"{type(exc).__name__}: {exc}"})
            raise
        try:
            path = Path(result)
            if path.suffix == ".json" and path.exists():
                inner = json.loads(path.read_text()).get("status")
                if inner in ("SKIPPED", "FAIL", "REVIEW"):
                    steps[-1]["status"] = inner
        except (TypeError, ValueError, OSError, json.JSONDecodeError):
            pass

    call("slice_correspondence", correspondence)
    call("registered_geometry", geometry)
    call("psf_reprojection", reproject)
    call("common_mask", common_mask)
    call("central_component_mask", central_mask)
    call("all_stack_metrics", slice_metrics)
    call("metric_summary", summarize)
    call("native_roi", build_roi)
    call("roi_final_pose", map_roi)
    call("roi_geometry", validate_roi)
    call("myocardial_metrics", myocardial)
    call("bland_altman", bland_altman)
    call("aha_masks", build_aha)
    call("aha_stats", aha_stats)
    call("aha_validation", validate_aha)
    call("figure1_candidates", candidates1)
    call("figure2_candidates", candidates2)
    if not quick:
        call("individual_qc", individual)
        call("individual_mapping_separate", lambda current_cfg, current_overwrite:
             separate_mapping(current_cfg, output_root(current_cfg), current_overwrite))
        call("individual_mapping_rows_shared_bar", lambda current_cfg, current_overwrite:
             mapping_rows(current_cfg, output_root(current_cfg), current_overwrite))
    call("metric_plot", metric_plot)
    call("roi_qc", roi_qc)
    call("aha_qc", aha_qc)
    call("ba_plot", ba_plot)
    call("bullseye_plot", bull_plot)
    overall = "ROI_PENDING" if any(
        x["status"] == "SKIPPED" and x["step"] in ("native_roi", "myocardial_metrics", "aha_masks")
        for x in steps
    ) else "COMPLETE"
    status = output_root(cfg) / "manifests" / "subject_evaluation_status.json"
    dump_json({"subject_id": cfg["subject_id"], "status": overall, "steps": steps},
              status, overwrite)
    return status

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True);p.add_argument("--overwrite",action="store_true");p.add_argument("--quick",action="store_true",help="Skip all per-slice PNG figures")
 a=p.parse_args();print(run(load_yaml(a.config),a.overwrite,a.quick))
if __name__=="__main__":main()
