# G0 result revision policy

`G0_geometry_correspondence_baseline_v1` is preserved. Its map correspondence audit is valid: all six T1/T2-stack combinations had diagonal assignment fraction 1.0, zero non-diagonal matches, and zero non-identity orientations. Its pose-derived outputs are invalid because G0 v1 inverted the vendored NeSVoR axis-angle component semantics.

`G0_geometry_correspondence_baseline_v2` is the corrected formal rerun target. It retains `code10w_g0_geometry_audit/v1` as the scientific result schema and records `implementation_revision: G0_geometry_correspondence_baseline_v2`. It changes only pose-derived outputs; correspondence, official expected pairing, and G0 signal behavior are unchanged.
