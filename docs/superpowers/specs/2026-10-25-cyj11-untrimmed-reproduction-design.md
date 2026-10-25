# CYJ 1.1 untrimmed-input reproduction design

## Intent

Reproduce the historical CYJ Experiment 1.1 route before extending it to the
other four subjects.  The pilot covers the complete historical CYJ matrix:
L100, both L50 folds, all four L25 folds, and the five sparse profiles
(seven resolved case/fold runs).

## Evidence and boundary

The historical CYJ source contains 14 SAX, 15 2CH, and 12 4CH native groups.
The current Final5 `02_standardized_12_15_10` source contains 12, 15, and 10
groups respectively, with different observation hashes.  It must not be used
for this reproduction.

This is a new immutable experiment root.  It does not modify or overwrite
Final5 reconstructions, endpoints, metrics, sparse manifests, or native
bundles.

## Chosen approach

Use the untrimmed prepared source for each subject and create new sparse
profiles.  Pin each historical stage rather than treating current Final5 as a
drop-in replacement:

- sparse-profile generator commit `6f4142cccc17782ba6e985e74531822560b369f3`;
- Q002S640/K8 reconstruction commit `de41aef1c52b12037efed20f71cdf0db4696d49c`;
- Fit-first output-sampling behaviour represented by commit
  `00d88b45c75f272f05538997e38fbda85988e7aa`.

Q002 uses the historical K8 training route and its historical default raw
exports (central and PSF128).  The reproduction publishes and evaluates only
the central formal endpoint; it does **not** pass a newer `central_only`
training/export option.  Q005/K4 is out of scope because it is a different
proposed method.

Fit-first reproduces the historical sparse behaviour: independent T1/T2
NeSVoR, and, for every sparse profile, the parameter-matched SAX
`stack.nii.gz` is passed to `--sample-mask`.  This mask is recorded as raw
NeSVoR output-sampling provenance only; it is not a training, registration,
or endpoint-query mask.

Central Fit-first endpoints query each sparse checkpoint with the frozen L100
checkpoint mask.  Evaluation reuses the historical stratified native
common-support semantics, producing only new outputs.

## Alternatives considered

- Current Final5 standardized 12/15/10 inputs: rejected because slice counts,
  group identities, and observation hashes differ from CYJ 1.1.
- Q005/K4: rejected because it changes the proposed training method.
- Current `fit_first_exp1_1.py`: rejected for strict reproduction because its
  single-stack binary-mask policy differs from the historical all-sparse SAX
  stack-image sampling policy.

## Admission and verification

Before any GPU command, validate each untrimmed subject source, generated
sparse manifest, frozen B6/decoder/protocol hashes, stage commit, and fresh
output root.  The first CYJ validation case is
`SAX100_2CH100_4CH0`; its dry-run must reproduce the historical command
contract for both Proposed and Fit-first.  After that validation, the same
route can be used for the remaining CYJ cases.  Observed wall time from the
Proposed and both NeSVoR logs then budgets the four-subject expansion.  No
server reconstruction is started by this design document.
