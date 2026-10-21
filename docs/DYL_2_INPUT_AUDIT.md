# DYL_2 input audit

Audit source: `/home/universe/SVR/multimap_postprogramming/subject_dicom/DYL_2`, read locally on 2026-10-21 with pydicom; no pixel data was altered.

| stack | readable DICOM | spatial groups x weights | rows x columns | pixel spacing mm | slice thickness mm | TR ms | series UID count |
|---|---:|---:|---|---|---:|---:|---:|
| sax | 120 | 12 x 10 | 288 x 256 | 1.25 x 1.25 | 8 | 2.61 | 1 |
| 2ch | 160 | 16 x 10 | 288 x 256 | 1.25 x 1.25 | 8 | 2.61 | 1 |
| 4ch | 100 | 10 x 10 | 256 x 288 | 1.25 x 1.25 | 8 | 2.61 | 1 |

All 380 files were readable. Each spatial position contains exactly 10 images; `AcquisitionTime`, `ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing`, `SliceThickness`, `RepetitionTime`, and `SeriesInstanceUID` are present and internally consistent per stack. The Siemens private protocol fields include `gre_bmultimap_test1_87` and a private value `87`, consistent with VPS=87, but this is not yet an independent decoder-ready protocol proof.

## Admission decision: BLOCKED_PENDING_PROTOCOL_TIMING_VALIDATION

The standard `InversionTime` field contains mixed/nonphysical values and does not directly prove the frozen TI `[50,150]` ms or T2prep `[35,45,55]` ms schedule. Therefore the audit does not assert protocol compatibility from a filename or private label. Before formal preprocessing, run the MATLAB preprocessing plus Python bridge in a new output location and verify its timing manifest, TR=2.61 and VPS=87 against the HHZ v1 contract. Any mismatch is a fail-closed rejection: do not alter timings, retrain B6/decoder, or route DYL_2 into a formal run.

`DYL_2` is a new subject ID. Its versioned local/server paths are declared in `configs/subject_stack_manifest_v2.json`; no DYL path is reused.
