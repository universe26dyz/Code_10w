# Module 05B — offline HHZ signal surrogate

This module is an offline, fixed-protocol surrogate only.  Its teacher is the
direct PyTorch port of HHZ `sim_T1T2_10HB_bssfp.m` in `trad_teacher/`; it is
not EPG, a dictionary/inverse model, or a subject-specific decoder.

`build_timing_pool.py` accepts explicitly supplied prepared observation NPZs,
requires every source group to have all ten weights, and rejects unequal TR or
VPS.  It writes unique `timing9_ms` rows and their source/group provenance.
`generate_mlp_dataset.py` generates `train.h5`, `valid.h5`, and `test.h5`
once, splitting by unique timing vector rather than by samples.  Samples use
T1 in [20,2500] ms, T2 in [5,200] ms with T1>T2, and B1 in [0.1,1.2].

The mDM-style network consumes
`[T1/1000, T2/1000, B1, timing2..10/1000]` and is exactly
`12 → (Linear(200), BN, LeakyReLU) × 3 → Linear(10)`.  Its raw output is L2
normalized for training and online use.  `FrozenMLPSignalDecoder` freezes
parameters and keeps BatchNorm in evaluation mode while preserving gradients
through T1/T2/B1; it never uses `torch.no_grad()` in `forward`.

The smoke script may create an explicitly labelled functional three-timing
fixture only when its single real timing vector cannot form train/valid/test
splits.  That checks the code path only and makes no cross-timing
generalization claim.  This phase deliberately does not connect the decoder
to reconstruction or replace the Trad forward model.
