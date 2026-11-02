# Pipeline contract

The pipeline is a server-first, three-stack workflow. Every stage receives an explicit project root and subject id; no stage may infer a patient from a filename or use the historical `Code_10w` absolute path.

The immutable timing contract is TR 2.61 ms, VPS 87, TI `[10, 100]` ms, T2-preparation `[35, 45, 55]` ms, flip angles `[45, 45, 45]`, and ten weights per group. `L100_fold0` consumes the full prepared subject root; profile cases consume their sparse prepared root.

The required completion order is `01` through `09`. A stage is complete only when its manifest and SHA/provenance record exist and its log contains the documented `PASS` marker. Raw DICOM, masks, models, MATLAB outputs, reconstructions and metrics stay outside Git tracking; their identities are recorded in manifests.

