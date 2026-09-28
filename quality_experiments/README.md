# Code_10w quality experiments

This directory contains read-only baseline diagnostics (D1–D3) and later reconstruction-quality experiments. It is intentionally separate from immutable baseline source directories (`trad/`, `mlp/`, `reconstruction_core/`, `config/`, and `docs/`) and from historical baseline outputs.

D1 separates direct native central-plane INR maps from map-domain parameter-PSF smoothing. D2 evaluates paired K=8 signal-domain agreement. D3 bridges PSF-integrated signals to the exact native dictionary fit. D1–D3 may compare Bloch and FrozenMLP because they diagnose the baseline. Subsequent quality experiments are FrozenMLP-only by default: the approved decoder remains frozen/eval-mode with decoder parameters `requires_grad=False`; only reconstruction-side variables may change.

All server results live below `/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/quality_experiments_v1/<SUBJECT_ID>/`. Every result root is self-contained and rejects a non-empty output directory. Code is named by phase/experiment; result directories use `D1_*`, `D2_*`, `D3_*`, then `Q001_*`, `Q002_*`, and so on. The append-only `EXPERIMENT_LOG.md` records commands, provenance, environment, result status, and interpretation. Local CPU work is limited to static checks, synthetic tests, smoke tests, and CLI validation; server-only outputs are never fabricated.
