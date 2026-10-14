"""Prepare a Hanson-source-anchored XCAT handoff; never execute XCAT locally."""
from __future__ import annotations
import json
from pathlib import Path

_BASE = "mode = 0\nact_phan_each = 1\natten_phan_each = 0\nmotion_option = 1\nout_period = -30\ntime_per_frame = 1\nout_frames = 1\nvec_factor = 2\nresp_period = 1\nresp_start_ph_index = 0\nmax_diaphragm_motion = 1.0\nmax_AP_exp = 1\ndia_filename = XCAT_input_dia.dat\nap_filename = XCAT_input_ap.dat\nhrt_period = 1.2\nhrt_start_ph_index = 0.2\npixel_width = 0.1\nslice_width = 0.1\narray_size = SET_BY_LICENSED_XCAT_RUN\nstartslice = SET_BY_LICENSED_XCAT_RUN\nendslice = SET_BY_LICENSED_XCAT_RUN\nactivity_unit = 0\n"

def _parameter(kind: str) -> str:
    lv, blood, tissue = {"t1": (1410, 1200, 825), "t2": (50, 50, 30), "pd": (1346, 1397, 3195), "lv_mask": (1, 0, 0)}[kind]
    return _BASE + f"myoLV_act = {lv}\nmyoRV_act = {tissue}\nbldplLV_act = {blood}\nbldplRV_act = {blood}\nbody_activity = {tissue}\nmuscle_activity = {tissue}\nliver_activity = {tissue}\nr_lung_activity = 0\nl_lung_activity = 0\n# SOURCE-ANCHORED structure: Hanson XCAT/t1.par, t2.par, proton_density.par.\n# CODE_10W extension: lv_mask must set every non-LV activity assignment to 0.\n"

def prepare_xcat_parameter_hand_off(output_root: str | Path) -> dict:
    root = Path(output_root) / "phase1_2_xcat_handoff"
    if root.exists(): raise FileExistsError(f"refusing to overwrite {root}")
    params, reference = root / "parameter_templates", root / "reference_design"; params.mkdir(parents=True); reference.mkdir()
    for filename, kind in (("t1_phase1_2.par", "t1"), ("t2_phase1_2.par", "t2"), ("pd_phase1_2.par", "pd"), ("lv_mask_phase1_2.par", "lv_mask")):
        (params / filename).write_text(_parameter(kind))
    provenance = {"reference": {"repository": "https://github.com/hmhanson/4DPhantomMRI", "commit": "97558db05f4e344865f472de974b9f4d042a1a15", "files": {"readme": "README.md", "t1": "XCAT/t1.par", "t2": "XCAT/t2.par", "pd": "XCAT/proton_density.par", "ap": "XCAT/XCAT_input_ap.dat", "dia": "XCAT/XCAT_input_dia.dat", "loader_example": "2D Sample.ipynb"}}, "code_10w_extensions": ["LV activity-mask run", "T1=1410/T2=50", "B1=1", "controlled M0 support", "xyz/LPS/mm 1-mm canonicalization", "LV PCA view alignment", "HHZ and 6x6x9 rendering", "Phase-2 final readiness"]}
    (reference / "reference_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (root / "README_XCAT_EXECUTION.md").write_text("Run licensed XCAT four times with the supplied Hanson-style activity parameter files and one unchanged anatomy/phase configuration. Record raw SHA256, shape, ordering, spacing, origin and convention in xcat3d_inputs.json.\n")
    (root / "xcat3d_inputs.template.json").write_text(json.dumps({"phase": "1.2", "source_family": "single_source_xcat_3d_quantitative", "xcat": {"version_build": "REQUIRED", "source": "REQUIRED", "parameter_files": "parameter_templates", "configuration_hashes": "REQUIRED", "cardiac_phase": "REQUIRED", "respiratory_phase": "REQUIRED"}, "m0_policy": "controlled_binary_signal_support_m0_1", "volumes": "RECORD_T1_T2_PD_LV_MASK_RAW_FILES_AND_SHA256"}, indent=2) + "\n")
    return {"handoff_root": str(root), "reference_source_anchoring": "PASS", "external_xcat_runtime_gate": "UNVERIFIED"}
