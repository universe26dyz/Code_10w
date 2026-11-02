# D2 formal K=8 signal-domain evaluation

`D2_BASELINE_SIGNAL_K8_V1` loads final Bloch and FrozenMLP reconstruction checkpoints read-only, restores their stored training space, calls the established native-plane reprojection path with exactly `n_samples=8`, and evaluates native weighted signal. It does not create an optimizer, train, rerun reconstruction, alter checkpoints, or write to baseline run folders.

All D2 artifacts are named `signal_reprojection_<stack>_K8.npz` and retain observed/predicted/residual arrays plus group, weight, timing, mask, TR, and VPS metadata. The manifest records K=8, seed `20260911`, source checkpoint and prepared-input SHA256 values, decoder provenance, protocol path, and Git commit.

Method-specific metrics use each method's own finite valid support and remain secondary diagnostics. Primary decoder comparisons, figures, global/per-stack/per-weight summaries, and best/worst weight use the stricter support `Bloch valid AND FrozenMLP valid AND finite observed AND finite both predictions`. Signal metrics are produced by `signal_agreement_metrics()` in `original_input_intensity`; SSIM is aggregated from 2-D observation-level values and is never represented as a pooled 3-D SSIM.

Run the formal server command only on `cr_dreme`:

```bash
bash quality_experiments/server_commands/run_D2_CYJ.sh
```

It checks CUDA, final checkpoints, either documented prepared-input layout, and a non-empty output root before running. Its output is `/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/quality_experiments_v1/CYJ/D2_k8_signal_domain_baseline_v1/`.
