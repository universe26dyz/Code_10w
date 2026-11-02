# 01 Raw DICOM

This module is the read-only input gate. It verifies SAX, 2CH and 4CH DICOM directories before MATLAB preprocessing.

`run_stage.sh` creates the single-subject manifest and checks the three DICOM counts.

Inputs are three server-resident DICOM directories, with exactly ten weights per group. The output is one single-subject deployment manifest plus a count log. Run it on the server and keep raw data outside GitHub.

```bash
PROJECT_ROOT=/path/to/ismrm_v2_1
SUBJECT=HHZ
RAW_ROOT=/data/.../mapping_subject_dicom
python "$PROJECT_ROOT/common/scripts/extract_subject_deployment_manifest.py" \
  --source "$PROJECT_ROOT/configs/subject_stack_manifest.example.json" \
  --subject "$SUBJECT" --output "$PROJECT_ROOT/subjects/$SUBJECT/manifest.json"
```

`local_dicom_dir` and `server_dicom_dir` must resolve to the same server path. Do not substitute NIfTI: SOP UID, acquisition timing and DICOM geometry are required downstream.
