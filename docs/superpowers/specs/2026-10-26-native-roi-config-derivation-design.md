# Native ROI Config Derivation Design

## Intent

Generate one immutable, subject-specific ROI derivation config from an existing
four-file manual SAX ROI source and that subject's unified native SAX cache.
This removes error-prone hand entry of one native-group to manual-slice mapping
per plane while preserving the current no-resampling ROI workflow.

## Constraints

- Downstream only: no reconstruction, formal metric, Final5, or legacy-pipeline
  changes.
- Use only `prepared/<subject>/sax/observations.npz`,
  `native_cache/<subject>/native_maps/sax`, and the declared four ROI NIfTIs.
- Every native plane must map exactly to one in-bounds manual ROI slice through
  an integer signed-permutation lattice; interpolation is forbidden.
- Generated configs are immutable and record SHA256 provenance for the prepared
  SAX artifact.
- The existing `derive-roi` command remains the independent second validator
  and bundle writer.

## Chosen approach

Add `derive_cyj11_native_roi_config(analysis_root, subject, source_roi,
output)` and CLI mode `derive-roi-config`.  It loads the native SAX cache,
checks the manual ROI triad and composite geometry using existing helpers, and
derives exactly one manual slice index for each native group.  It writes the
existing `cyj11_native_roi_derivation_config/v1` schema, so no downstream
format changes are needed.

The command rejects ambiguous, non-integral, clipped, or duplicate mappings.
It is subject-generic; no subject ID or case name is hard-coded.
