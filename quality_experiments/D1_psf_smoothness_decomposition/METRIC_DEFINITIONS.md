# D1 metric definitions

All metrics are calculated on the original native pixel grid. No resizing, padding, or metric-time cropping is performed. A display crop is used only for figures.

`N`, bias, MAE, RMSE, NRMSE, Pearson, NCC, and SSIM reuse `trad.evaluation.scmr.metrics.agreement_metrics`. NRMSE is RMSE divided by the native in-support range. SSIM uses the existing support-bounded implementation, with invalid pixels filled by the relevant in-support mean only inside its temporary SSIM bounding box.

For a prediction mode, method-specific support is `native_valid AND finite(native) AND mode_export_support AND finite(prediction)`. The D1 strict paired support is `native_valid AND finite(native) AND central_bloch_support AND finite(central_bloch) AND psf_bloch_support AND finite(psf_bloch) AND central_mlp_support AND finite(central_mlp) AND psf_mlp_support AND finite(psf_mlp)`.

Gradient magnitude uses central differences in row and column directions. A pixel is eligible only when it and all four axial neighbours are in the paired support. `gradient_magnitude_ratio = mean(|grad prediction|) / mean(|grad native|)`. `gradient_correlation` is Pearson correlation of paired native and prediction gradient magnitudes. The edge mask is eligible native-gradient pixels at or above the 75th percentile of the eligible native-gradient distribution; `edge_RMSE_ms` is the prediction-versus-native RMSE on that mask.

High-frequency energy is deterministic: crop to the support bounding box; replace out-of-support pixels with the in-support mean; subtract that mean; multiply by a separable 2-D Hann window; take the 2-D FFT and squared magnitude; sum power where radial frequency is at least 0.25 cycles/pixel. `high_frequency_energy_ratio = HF(prediction) / HF(native)`.

`PSF−central` values are direct metric differences. They are calculated both on a within-method central/PSF common support and on the four-way strict paired support, so an apparent increment cannot result from changing valid pixels.
