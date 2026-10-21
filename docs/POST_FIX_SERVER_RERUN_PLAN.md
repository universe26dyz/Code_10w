# Post-fix server rerun plan

No reconstruction rerun is required. Do not launch NeSVoR, Fit-first reconstruction, Q002, or Q004 for this fix.

After the code commit is deployed, CYJ needs only read-only/evaluation outputs regenerated from already completed endpoint exports:

1. Generate immutable subject figure configs with `build_final_subject_figure_configs.py` using the completed formal grid, sparse manifests, R0 native cache, and current-native masks.
2. Run `final_stratified_native_metrics.py` for each completed condition/endpoint and proposed method against its matched Fit-first endpoint.
3. Regenerate figures from the generated configs.
4. Aggregate into a new v4 summary directory with `finalize_subject_reporting.sh`; do not overwrite an existing `final_cyj_v2` or another immutable output.

Every command must target a new output path. Check endpoint, subject, condition, manifest SHA, and current-native mask provenance before publication.
