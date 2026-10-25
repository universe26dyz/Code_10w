# CYJ 1.1 untrimmed sparse-case reproduction

This runbook is for a new immutable validation root.  It does not use the
Final5 `02_standardized_12_15_10` inputs and must never point at an existing
Final5 root.

The first validation is `CYJ/SAX100_2CH100_4CH0`.  It reconstructs the sparse
Proposed and Fit-first routes and then publishes/evaluates only `central`.
The historical Q002 invocation retains its default raw output behaviour; this
is not a new `central_only` training flag.  Fit-first retains the historical
parameter-matched SAX `stack.nii.gz` output sampling mask for both T1 and T2.

## One-time server setup

Use detached worktrees, rather than checking out historical commits in the
current server checkout:

```bash
CODE=/data/dengyz/code/Code_10w
WORK=/data/dengyz/code/Code_10w_historical_worktrees
mkdir -p "$WORK"
git -C "$CODE" worktree add --detach "$WORK/sparse-6f414" 6f4142cccc17782ba6e985e74531822560b369f3
git -C "$CODE" worktree add --detach "$WORK/q002-de41" de41aef1c52b12037efed20f71cdf0db4696d49c
git -C "$CODE" worktree add --detach "$WORK/fit-00d88" 00d88b45c75f272f05538997e38fbda85988e7aa
```

The following checks must report `14`, `15`, and `12`.  A Final5 12/15/10
source is deliberately rejected by the wrapper.

```bash
PY=/home/dengyz/miniconda3/envs/cr_dreme/bin/python
PREP=/data/dengyz/dataset/Code_10w_v1/Code_10w_prepared
"$PY" - "$PREP" <<'PY'
import sys
from pathlib import Path
import numpy as np
for stack in ("sax", "2ch", "4ch"):
    with np.load(Path(sys.argv[1]) / "CYJ" / stack / "observations.npz", allow_pickle=False) as x:
        print(stack, len(np.unique(x["group_idx"])))
PY
```

For the first sparse-case validation the historical L100 Fit-first root is an
explicit verified dependency: its learned poses are required to publish the
sparse Fit-first central endpoint.  The sparse Proposed endpoint is produced
from the selected sparse Q002 reconstruction, so an L100 Proposed root is not
an input and is deliberately not required.  This wrapper does not regenerate
L100.  Check the required dependency before use:

```bash
DATA=/data/dengyz/dataset/Code_10w_v1
R0_FIT="$DATA/Code_10w_runs/paper_joint_vs_2dfit/real/R0_full/R0_PHASE4R/fit_first/CYJ"
R0_CACHE="$DATA/Code_10w_runs/paper_joint_vs_2dfit/real/R0_native_cache/R0_PHASE4R/CYJ/native_maps"
NATIVE_MASK="$DATA/Code_10w_runs/paper_joint_vs_2dfit/formal_native_masks/CYJ_L100_fold0_current_native_v1"
test -f "$R0_FIT/nesvor/t1/model.pt" && test -f "$NATIVE_MASK/manifest.json"
```

## CYJ validation dry-run

Set the dependencies and run the wrapper without `--execute`.  It checks the
three worktree commits, input group integrity, B6/protocol hashes, L100 roots,
native-mask manifest, and target freshness; it creates no output directory.

```bash
export REPO=/data/dengyz/code/Code_10w
export CYJ11_PYTHON=/home/dengyz/miniconda3/envs/cr_dreme/bin/python
export CYJ11_PREPARED_ROOT=/data/dengyz/dataset/Code_10w_v1/Code_10w_prepared
export CYJ11_SPARSE_WORKTREE=/data/dengyz/code/Code_10w_historical_worktrees/sparse-6f414
export CYJ11_Q002_WORKTREE=/data/dengyz/code/Code_10w_historical_worktrees/q002-de41
export CYJ11_FIT_FIRST_WORKTREE=/data/dengyz/code/Code_10w_historical_worktrees/fit-00d88
export CYJ11_B6_MODEL=/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/runs/frozen_mlp_B6/model.pt
export CYJ11_SIGNAL_SIMULATOR=/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/inputs/mlp/signal_simulator_approved.pth
export CYJ11_PROTOCOL="$CYJ11_Q002_WORKTREE/trad/configs/protocol_hhz_v1.yaml"
export CYJ11_EXPECTED_B6_SHA256=15981fbe93d2c70e14755ea756a02856c9383481e7d92d239619268a628a29b5
export CYJ11_EXPECTED_SIGNAL_SIMULATOR_SHA256=f83280d4f73bfc9e0540b9606ee20bf83395339ad4e54dd0fb7be93ed9c9a7a9
export CYJ11_EXPECTED_PROTOCOL_SHA256=5b6bcf41f7766b29808d664f18b6def8894696ffb4ae3a55cfcba5532cdea5c3
export CYJ11_NATIVE_CACHE_ROOT="$R0_CACHE"
export CYJ11_NATIVE_MASK_ROOT="$NATIVE_MASK"
export CYJ11_L100_FIT_FIRST_ROOT="$R0_FIT"
export CYJ11_NESVOR_EXECUTABLE=/absolute/path/to/the/historical/nesvor
export CYJ11_OUTPUT_ROOT="$DATA/Code_10w_runs/cyj11_untrimmed_validation_20261025/SAX100_2CH100_4CH0"

bash "$REPO/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_cyj11_untrimmed_reproduction.sh" --dry-run
```

Inspect the printed plan before execution.  It must show the three commit
pins, Q002S640/K8 with anchor batch `640`, no `--endpoint-policy`, and
`nesvor_inputs/sax/t1/stack.nii.gz` plus
`nesvor_inputs/sax/t2/stack.nii.gz` as expected output-sampling masks.

## Execute and compare

Only after the dry-run is accepted, execute the same command with `--execute`:

```bash
bash "$REPO/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_cyj11_untrimmed_reproduction.sh" --execute
```

The new root stores `reproduction_plan.json` and `commands.log`.  Compare it
with the historical case using: sparse-manifest identity/hash, Q002 experiment
manifest, Fit-first `input_manifest.json` output-sampling provenance, central
endpoint manifests, and the final `aggregate_metrics.json` digest/content.
Do not compare a changed endpoint's RMSE as proof of identity; the command and
provenance contracts are the primary reproduction checks.

## Other-subject endpoint validation

The same planner also supports an explicit full-input endpoint-only pilot for
`DYZ`, `DYL_2`, `HHZ`, or `HJL`.  This is not a claim that those subjects have
CYJ's 14/15/12 layout: the plan records their actual complete ten-weight group
counts and the caller supplies every subject-specific root.  Metric generation
is deliberately disabled until the Final5 native-mask schema is admitted
separately.

The historical sparse profile filenames retain a `CYJ_` prefix, but their JSON
payload is subject-neutral (`SAX100_2CH100_4CH0`: SAX/2CH all, 4CH none).  The
generated sparse manifest still records and validates the explicit subject.

Use the audited full root for the selected subject, an explicit L100 Fit-first
root, and the subject's existing native-map cache.  Do not substitute another
subject's root and do not point this at `02_standardized_12_15_10`.

```bash
export CYJ11_SUBJECT=HHZ
export CYJ11_CASE=SAX100_2CH100_4CH0
export CYJ11_PREPARED_ROOT="$DATA/Code_10w_runs/phase4r_final_five_subject_k8_central_20261023/01_inputs_full/prepared"
export CYJ11_L100_FIT_FIRST_ROOT="$DATA/Code_10w_runs/phase4r_final_five_subject_k8_central_20261023/04_reconstruction/fit_first_exp1_1/HHZ/L100"
export CYJ11_NATIVE_CACHE_ROOT=/explicit/audited/HHZ/native_maps
export CYJ11_SKIP_METRIC=1
unset CYJ11_NATIVE_MASK_ROOT
export CYJ11_OUTPUT_ROOT="$DATA/Code_10w_runs/subject_full_endpoint_validation_20261026/$CYJ11_SUBJECT/$CYJ11_CASE"

bash "$REPO/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_cyj11_untrimmed_reproduction.sh" --dry-run
```

For `DYL_2`, the current L100 Fit-first root is under `fit_first/DYL_2/L100`;
the other audited subjects use `fit_first_exp1_1/<SUBJECT>/L100`.  Inspect the
dry-run plan before `--execute`: it must say
`subject_full_input_audited`, show the subject's actual counts, omit
`metric_central`, and retain only the Proposed/Fit-first central endpoints.

## Remaining CYJ matrix

After this case succeeds, run a dry-run per remaining sparse case using a
separate fresh output root, then execute serially on one GPU:

```bash
for CYJ11_CASE in L50_fold0 L50_fold1 L25_fold0 L25_fold1 L25_fold2 L25_fold3 \
  SAX100_2CH0_4CH0 SAX100_2CH0_4CH100 SAX100_2CH100_4CH0 \
  S2CH50_4CH100_fold0 S2CH50_4CH100_fold1 S2CH100_4CH50_fold0 S2CH100_4CH50_fold1; do
  export CYJ11_CASE
  export CYJ11_OUTPUT_ROOT="$DATA/Code_10w_runs/cyj11_untrimmed_validation_20261025/$CYJ11_CASE"
  bash "$REPO/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_cyj11_untrimmed_reproduction.sh" --dry-run
done
```

The L100 reproduction itself is a separate future admission path because this
validation wrapper intentionally requires existing L100 dependencies.  It
does not replace, overwrite, or re-run those L100 outputs.
