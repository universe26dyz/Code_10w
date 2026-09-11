function opts = preprocess_options_v1(max_slices, enable_figures, enable_mppca)
%PREPROCESS_OPTIONS_V1 Build explicit options for preprocess_stack_v1.
%   Callers choose each top-level switch explicitly. The MP-PCA parameters
%   below are the vendored DYZ routine's documented Phase-1 settings.

    arguments
        max_slices
        enable_figures (1, 1) logical
        enable_mppca (1, 1) logical
    end

    opts = struct();
    opts.max_slices = max_slices;
    opts.enable_figures = enable_figures;
    opts.enable_mppca = enable_mppca;
    opts.mind_alpha = 0.3;
    opts.mppca_options = struct( ...
        'patch_size', [7, 7], ...
        'center_data', true, ...
        'clip_negative', true, ...
        'use_parallel', false, ...
        'mask_threshold_fraction', 1/5, ...
        'min_component_size', 20, ...
        'fill_holes', false);
end
