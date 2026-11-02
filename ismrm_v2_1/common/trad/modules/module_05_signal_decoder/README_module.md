# Module 05A — Trad HHZ signal decoder

`TradSignalSimulator` is a direct PyTorch translation of
`hhz_original/Utility/sim_T1T2_10HB_bssfp.m`: ten heartbeats, perfect
inversions, T1 recovery, T2-prep, B1-scaled FA, ten ramp-up pulses,
alternating directions, 2×2 `Mxy/Mz` rotations, E1/E2 decay, the HHZ central
16-readout average, and carry-over Mz.  `timing9_ms` is
`Duration_befor_Acq[1:]`; heartbeat 1 uses a duration of zero.

No EPG implementation is used or provided.  The downstream forward model
uses the normalized fingerprint multiplied by INR amplitude `A`.
