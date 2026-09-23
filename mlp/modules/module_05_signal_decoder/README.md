# Module 05B — offline HHZ signal surrogate

This module is an offline, fixed-protocol surrogate only.  Its teacher is the
direct PyTorch port of HHZ `sim_T1T2_10HB_bssfp.m` in `trad_teacher/`; it is
not EPG, a dictionary/inverse model, or a subject-specific decoder.

`generate_rr_mlp_dataset.py` is the only active dataset entry.  It generates
rhythm-disjoint `train.h5`, `valid.h5`, and `test.h5` directly from the fixed
HHZ/VPS=87 RR synthetic domain; it does not read prepared subject observations.
Samples use T1 in [20,2500] ms, T2 in [5,200] ms with T1>T2, and B1 in
[0.1,1.2].

The mDM-style network consumes
`[T1/1000, T2/1000, B1, timing2..10/1000]` and is exactly
`12 → (Linear(200), BN, LeakyReLU) × 3 → Linear(10)`.  Its raw output is L2
normalized for training and online use.  `FrozenMLPSignalDecoder` freezes
parameters and keeps BatchNorm in evaluation mode while preserving gradients
through T1/T2/B1; it never uses `torch.no_grad()` in `forward`.

Dataset metadata and the checkpoint record the RR protocol, rhythm split, and
timing domain.  Subject/timing-pool workflow sources are retained under
`archive/legacy_subject_timing_mlp/`, but are not active import or CLI paths.
