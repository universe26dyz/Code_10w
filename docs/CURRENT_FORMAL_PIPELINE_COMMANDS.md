# Current formal pipeline commands

All commands below are templates. Replace only the explicitly marked run/output paths; existing formal outputs are immutable and every producer must refuse overwrite. Local commands are CPU-only and do not invoke CUDA.

## LOCAL: preprocessing and bridge

```bash
cd /home/universe/SVR/multimap_postprogramming/Code_10w
matlab -batch "addpath('trad/scripts'); local_preprocess_all('configs/subject_stack_manifest_v2.json','/home/universe/SVR/data/Code_10w_v1/Code_10w_preprocessed')"
conda run --no-capture-output -n knesvr_torch python -m trad.modules.module_02_data_bridge.prepare_observations \
  --preprocessed-mat /home/universe/SVR/data/Code_10w_v1/Code_10w_preprocessed/DYL_2/sax/preprocessed.mat \
  --dicom-dir /home/universe/SVR/multimap_postprogramming/subject_dicom/DYL_2/sax \
  --stack sax --output-dir /home/universe/SVR/data/Code_10w_v1/Code_10w_prepared/DYL_2/sax
```

Repeat the bridge command for `2ch` and `4ch`. Expected outputs are `observations.npz`, `timing.npy`, `manifest.json`, and `qc_summary.json`; inspect their TR/VPS and timing before transfer. Do not run the MATLAB command until `docs/DYL_2_INPUT_AUDIT.md` is cleared.

## Transfer and server validation

```bash
rsync -a --checksum /home/universe/SVR/data/Code_10w_v1/Code_10w_prepared/DYL_2/ \
  dengyz@SERVER:/data/dengyz/dataset/Code_10w_v1/Code_10w_prepared/DYL_2/
ssh dengyz@SERVER 'cd /data/dengyz/code/Code_10w && /home/dengyz/miniconda3/envs/cr_dreme/bin/python -m paper_experiments.joint_vs_2dfit.phase2_clean.sparse --prepared-root /data/dengyz/dataset/Code_10w_v1/Code_10w_prepared --subject DYL_2 --condition L100 --phase 0 --output-root /data/dengyz/dataset/Code_10w_v1/Code_10w_runs/paper_joint_vs_2dfit/sparse_inputs'
```

The sparse command is CPU/light-to-moderate and writes an immutable manifest. Create exactly L100/fold0, L50/fold0..1 and L25/fold0..3; do not add L33 to the paper grid.

## SERVER_GPU: Q002/Q004 and Fit-first

```bash
export REPO=/data/dengyz/code/Code_10w
export DATA=/data/dengyz/dataset/Code_10w_v1
export PY=/home/dengyz/miniconda3/envs/cr_dreme/bin/python
export NESVOR=/home/dengyz/miniconda3/envs/nesvor/bin/nesvor
export DECODER="$DATA/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/inputs/mlp/signal_simulator_approved.pth"
export B6="$DATA/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/runs/frozen_mlp_B6/model.pt"
```

For Q004, call the checked-in server wrapper with `SUBJECT=DYL_2`. The selected prepared root is the R0 root for L100 and the relevant sparse root for L50/L25; a nonempty destination fails before training.

```bash
SUBJECT=DYL_2 "$REPO/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_q004s640_cyj.sh" \
  L100_fold0 "$DATA/Code_10w_prepared/DYL_2" "$DATA/Code_10w_prepared/DYL_2" \
  "$DATA/Code_10w_runs/paper_joint_vs_2dfit/sparse_inputs/DYL_2/L100_fold0/sparse_manifest.json" \
  "$B6" "$DECODER" "$DATA/Code_10w_runs/paper_joint_vs_2dfit/real/DYL_2/Q004S640/L100_fold0"
```

Q002 uses the same prepared root/dependencies and its formal Q002S640 route. Both are GPU/expensive and must record dependency SHA256, input SHA256, git SHA, seed and route manifest. Fit-first first performs native dictionary matching, then invokes NeSVoR independently for `t1` and `t2` with the final poses audited before bundle export.

## SERVER_CPU: endpoint and reporting sequence

1. Build the R0 native cache and final-pose audit.
2. Export direct central/PSF128 endpoints and Fit-first central/PSF128 endpoints.
3. Run `current_native_central_component_mask` from R0 source support plus L100 support.
4. For L100, build direct and Fit-first final-pose native bundles, then run `final_pose_native_metrics`.
5. For L50/L25, run `final_sparse_native_metrics`; it only samples manifest held-out 2CH/4CH original native geometry and carries `no_learned_pose=True`.
6. Generate native mosaics, SCMR-style longitudinal figures, and a subject summary only from immutable metric artifacts.

Use the subject layout below for newly generated work; historical CYJ output locations remain readable.

```text
$EXP/formal_native_grid_v2/DYL_2_<RUN_TAG>/
  endpoint_exports/  native_bundles/  metrics/  figures/  configs/  summary/
```
