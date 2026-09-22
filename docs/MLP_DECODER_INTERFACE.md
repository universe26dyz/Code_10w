# MLP decoder interface

This document defines the decoder-only boundary for the controlled Trad versus
FrozenMLP experiment.  Neither decoder owns preprocessing, spatial PSF,
rigid registration, amplitude scaling, loss, or optimization.

## Inputs

Both decoders accept PyTorch floating tensors in physical units:

| Name | Shape | Unit | Meaning |
|---|---:|---|---|
| `t1_ms` | `[...]` | ms | Quantitative INR T1 field at PSF samples |
| `t2_ms` | `[...]` | ms | Quantitative INR T2 field at PSF samples |
| `b1` | `[...]` | relative | Quantitative INR transmit field |
| `timing9_ms` | `[..., 9]` | ms | HHZ durations before acquisitions 2–10 |
| `protocol` | scalar object | ms/count/degrees | DICOM TR/VPS plus fixed HHZ sequence constants |
| `normalize` | `True` | — | Request L2-normalized fingerprint |

Leading dimensions are batch dimensions and must be preserved. The ordering is
strict: the timing axis is heartbeat/acquisition 2 through 10, while the
decoder output axis is observed weight/acquisition 1 through 10.

## Output

Both decoders return `fingerprint [..., 10]`, a finite, L2-normalized
10-weight signal fingerprint with the same floating dtype as `t1_ms`.
`TradQuantitativeForward` selects one observed weight, multiplies the INR
amplitude outside this decoder boundary, and averages spatial PSF samples.

## Frozen MLP requirements

`FrozenMLPSignalDecoder` builds its 12 features as
`[T1/1000, T2/1000, B1, timing9/1000]`, runs its frozen BatchNorm MLP, and
returns the normalized fingerprint. Its parameters have `requires_grad=False`
and BatchNorm remains in evaluation mode. Autograd must remain enabled for
`t1_ms`, `t2_ms`, and `b1`; gradients flow through the frozen network to the
reconstruction parameters only.

Checkpoint compatibility is enforced exclusively by the existing loader.
Candidates not marked `approved_by_manual_review`, protocol mismatches, timing
extrapolation, and functional fixtures remain ineligible for formal online
reconstruction.
