# Upstream provenance

## NeSVoR

- URL: https://github.com/daviddmc/NeSVoR
- Branch: `master`
- Commit: `2e96a91bdd30174210caea911e03a2778c65adbe`
- Copied in Phase 1: `LICENSE`, `README.md`, and the complete pure-Python
  `nesvor/` package (only `*.py`, preserving source headers and import closure).
- Project-local modifications: none in Phase 1.

## Model-Based-MoCo-for-Cardiac-Multi-Parametric-Mapping (mDM reference)

- URL: https://github.com/SJTU-CMRLab/Model-Based-MoCo-for-Cardiac-Multi-Parametric-Mapping
- Branch: `master`
- Commit: `a24ab1008d48e92ddf6f2bb8131f9a58dda23e95`
- Copied in Phase 1: only
  `DM_LR_with_fast_dictionary_generation/Models/signal_simulation.py`,
  `train_ss_net.py`, and `test_ss_net.py`, under
  `mdm/fast_dictionary_generation_reference/`.
- Intended use: architecture/training/test reference only. Its EPG code is not
  copied or used as a project model.

No `.git` directory is vendored.
