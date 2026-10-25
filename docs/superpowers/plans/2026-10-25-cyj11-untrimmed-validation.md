# CYJ 1.1 untrimmed validation implementation plan

> Execute this plan as one local implementation pass.  Do not create
> worktrees, run GPU jobs, alter existing experiment outputs, or push a
> commit.  The server wrapper must default to dry-run.

**Goal:** Add a small, provenance-first command builder for an immutable,
strict CYJ 1.1 re-production using the untrimmed inputs, then make
`CYJ/SAX100_2CH100_4CH0` the first server validation of both Proposed and
Fit-first.

**Architecture:** A Python module owns canonical case identities, historical
commit pins, source/worktree admission checks, and deterministic command-plan
serialization.  A thin server shell wrapper supplies explicit paths and
either prints the plan or executes it.  It does not reuse Final5's current
runner or mutate its output tree.

**Tech stack:** Python standard library, existing Phase4R/Q002/Fit-first
entry points in pinned worktrees, pytest, bash.

**Spec:**
[`docs/superpowers/specs/2026-10-25-cyj11-untrimmed-reproduction-design.md`](../specs/2026-10-25-cyj11-untrimmed-reproduction-design.md)

## Global constraints

- The source is the untrimmed `Code_10w_prepared/CYJ` route, never Final5
  `02_standardized_12_15_10`.
- Pin sparse generation to `6f4142cccc17782ba6e985e74531822560b369f3`, Q002
  to `de41aef1c52b12037efed20f71cdf0db4696d49c`, and Fit-first/endpoint code
  to `00d88b45c75f272f05538997e38fbda85988e7aa`.
- Preserve the historical Q002 default raw output route.  Publish/evaluate
  central only after it; never pass a newer `central_only` option.
- Preserve historical Fit-first sparse `--sample-mask` use for all sparse
  cases: parameter-matched SAX `stack.nii.gz`, not `stack_mask.nii.gz`.
- New results are written below a caller-provided fresh experiment root.  The
  code must reject an existing non-empty target instead of overwriting it.
- This phase implements a CYJ compatibility route.  It deliberately does not
  fabricate a new-subject historical primary mask; four-subject metric
  expansion follows successful CYJ validation.

## Review focus

- The printed command plan must make every source, commit, hash, endpoint and
  output location explicit.
- The Proposed and Fit-first calls must reflect the historical command
  contract, including central-only publication (not central-only training).
- Admission failures must be actionable and must not silently substitute
  Final5 inputs or current code.

---

## Task 1: deterministic historical-route plan and admission checks

**Files:**

- Create `paper_experiments/joint_vs_2dfit/phase2_clean/cyj11_untrimmed_reproduction.py`
- Create `paper_experiments/joint_vs_2dfit/tests/test_cyj11_untrimmed_reproduction.py`

**Steps:**

1. Add failing tests using temporary untrimmed-style prepared inputs and
   synthetic git-worktree metadata for the following behaviours:
   - canonical CYJ case resolution, initially including
     `SAX100_2CH100_4CH0`;
   - rejection of a standardized Final5 prepared root and of the CYJ
     12/15/10 group layout;
   - acceptance only of the historical CYJ 14/15/12 layout for this strict
     compatibility route;
   - rejection of a worktree at the wrong commit or a non-empty output root;
   - generated Proposed command uses historical Q002S640/K8 settings and does
     not add `--endpoint-policy central_only`;
   - generated sparse Fit-first T1/T2 commands use SAX `stack.nii.gz` as
     `--sample-mask`, including a multi-stack case;
   - formalization/metric entries select only `central` for publication.
2. Implement small typed records/constants for the stage commits, the seven
   historical sparse case/fold identities, and legacy L100/L50/L25 identities.
   Keep all path construction explicit; do not recursively discover files.
3. Implement source and worktree validation.  Check each required
   `observations.npz`, its stack identity and group count, explicit SHA pins
   supplied by the caller where available, and `git rev-parse HEAD` of each
   supplied worktree.  Error messages must identify the offending stack/path.
4. Build a JSON-serializable plan with commands in dependency order:
   sparse input generation, L100 Proposed/Fit-first prerequisites, selected
   sparse Proposed/Fit-first case, central formal endpoint exports, and the
   historical central metric invocation.  Include source/output provenance and
   the historical native-mask input in the plan.
5. Run the new targeted pytest file and fix only failures caused by this work.

## Task 2: explicit server wrapper and runbook

**Files:**

- Create `paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_cyj11_untrimmed_reproduction.sh`
- Create `docs/CYJ_1_1_UNTRIMMED_REPRODUCTION_RUNBOOK.md`
- Update the targeted pytest file from Task 1 as needed

**Steps:**

1. Add a wrapper requiring explicit environment variables for the Python
   interpreter, untrimmed prepared root, B6/decoder/protocol inputs, historical
   native cache and mask, output root, and the three pinned worktrees.
2. Make dry-run the default.  It validates every path/commit and prints the
   serialized plan without making output directories or invoking a GPU.
   Require `--execute` for server execution; execute the already-built plan
   sequentially, retaining a command log under the new root.
3. Document the exact `CYJ/SAX100_2CH100_4CH0` dry-run and execution commands,
   the expected historical evidence to compare (manifests, endpoint schema,
   sample-mask provenance, aggregate metric SHA/content), and the subsequent
   full-CYJ matrix command.  Explicitly state that the wrapper must not be
   aimed at the existing Final5 root.
4. Add a static/CLI test that demonstrates dry-run has no output-side effects
   and commands contain only deterministic configured paths.

## Task 3: final local verification and handoff

**Files:** all files above only.

**Steps:**

1. Run the targeted pytest file, then the closest existing Phase2 tests that
   exercise sparse-profile and Fit-first command construction if they are
   available without GPU/server data.
2. Run:
   ```bash
   python -m compileall -q paper_experiments/joint_vs_2dfit/phase2_clean
   git diff --check
   ```
3. Inspect the diff and ensure it does not modify reconstruction algorithms,
   Final5 runners, formal outputs, or third-party code.
4. Create one local commit (no amend, no push), then report the exact commit
   SHA, local test results, and the server-only validation command.
