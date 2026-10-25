# Final5 single-element publication visualization

`final5_publication_visualization` is a read-only publication visualization consumer, not a quantitative evaluator. The sole authority for Final5 RMSE/MAE/bias remains `06_metrics/final_native_grid_v2/<SUBJECT>/<CASE>/` (`final5_native_grid_metric_bundle/v2`). It never trains, reconstructs, exports endpoints, matches dictionaries, or recomputes metrics.

It consumes deterministic Final5 paths: the sparse manifest under `03_sparse_inputs`, R0 native dictionary under `02_standardized_12_15_10/native_dictionary`, central Proposed and Fit-first endpoints under `05_formal_endpoints/central`, and the v2 formal metric bundle. All subject/case/schema/central endpoint and sparse/endpoint SHA provenance checks fail closed.

The immutable output root is `07_publication_visualization/single_element_v1/<SUBJECT>/<CASE>/`. A source plane is written as `T1/2ch/retained/group003/` (or `heldout`, `observed`) and contains five independent clean assets, each in PNG and PDF: `native_reference`, `proposed`, `fit_first`, `proposed_absolute_error`, and `fit_first_absolute_error`. There are no titles, axes, panel labels, or embedded colorbars. `assets/` supplies separate T1/T2 map and error colorbars.

For sparse cases, SAX is always retained as `observed`; 2CH/4CH `retained` and `heldout` names are taken exclusively from the sparse manifest's original-group lists. L100 marks every source plane `observed` and never invents held-out planes. The renderer loads every source group, so retained LAX and SAX are not lost merely because formal sparse metrics evaluate held-out LAX groups.

Native, Proposed, Fit-first, and both absolute-error assets share a single crop calculated from the same display support: `source_support AND finite(reference, proposed, fit_first) AND proposed_reprojection_valid AND fit_first_volume_geometry_valid`. This is deliberately broader than the formal metric ROI and performs no connected-component or myocardium-only filtering. Quantitative maps use the formal evaluator's R0 LPS-to-RAS affine and `reproject_volume_to_native_plane` (linear/order 1); binary masks retain nearest-neighbour/order 0 semantics. T1/T2 maps use the exact SCMR LUT and fixed ranges 0–2500/0–200 ms; errors use 0–300/0–30 ms.

Dry-run checks inputs and reports `READY`, `MISSING_INPUT`, `ALREADY_RENDERED`, or `BLOCKED_INVALID` without creating output:

```bash
cd /data/dengyz/code/Code_10w
SUBJECT=DYL_2 CASE=L100 \
  bash paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/render_final5_publication_visualization.sh --dry-run
```

Rendering uses the same command without `--dry-run`; an existing output directory is refused rather than overwritten. Optional `--parameter t1|t2`, `--stack sax|2ch|4ch`, and `--group N` filters only select assets and never modify the display or metric definitions.
