function local_preprocess_one(input_dicom_dir, output_mat_path, subject_id, stack_name)
%LOCAL_PREPROCESS_ONE Run the frozen v1 MATLAB preprocessing for one explicit stack.
%   This local-only entry never chooses a DICOM directory, subject, or output
%   path. It refuses overwrite through both this wrapper and preprocess_stack_v1.

    arguments
        input_dicom_dir (1, :) char
        output_mat_path (1, :) char
        subject_id (1, :) char
        stack_name (1, :) char
    end

    script_dir = fileparts(mfilename('fullpath'));
    method_root = fileparts(script_dir);
    addpath(fullfile(method_root, 'modules', 'module_01_preprocess_matlab', 'preprocessing_v1'));
    assert(isfolder(input_dicom_dir), 'local_preprocess_one:InputDirectory', ...
        'DICOM directory does not exist: %s', input_dicom_dir);
    output_dir = fileparts(output_mat_path);
    assert(~isempty(output_dir), 'local_preprocess_one:OutputDirectory', 'Output MAT must include a directory.');
    assert(isfolder(output_dir), 'local_preprocess_one:OutputDirectory', 'Output directory does not exist: %s', output_dir);
    assert(~isfile(output_mat_path), 'local_preprocess_one:OutputExists', 'Refusing to overwrite: %s', output_mat_path);
    qc_path = fullfile(output_dir, 'preprocess_qc.json');
    assert(~isfile(qc_path), 'local_preprocess_one:QCExists', 'Refusing to overwrite: %s', qc_path);

    opts = preprocess_options_v1([], false, true);
    preprocess_stack_v1(input_dicom_dir, output_mat_path, opts);
    saved = load(output_mat_path, 'Mag_crop', 'TR', 'VPS', 'preprocessing_options', 'mppca_center_data');
    required_fields = {'Mag_crop', 'TR', 'VPS', 'preprocessing_options', 'mppca_center_data'};
    for field_index = 1:numel(required_fields)
        assert(isfield(saved, required_fields{field_index}), 'local_preprocess_one:SavedMetadata', ...
            'Saved MAT lacks required metadata %s: %s', required_fields{field_index}, output_mat_path);
    end
    assert(size(saved.Mag_crop, 3) == 10, 'local_preprocess_one:Weights', ...
        'Saved Mag_crop must have exactly 10 weights in dimension 3.');
    saved_opts = saved.preprocessing_options;
    option_fields = {'enable_mppca', 'enable_figures', 'max_slices', 'mind_alpha', 'mppca_options'};
    for field_index = 1:numel(option_fields)
        assert(isfield(saved_opts, option_fields{field_index}), 'local_preprocess_one:SavedOptions', ...
            'Saved preprocessing_options lacks %s.', option_fields{field_index});
    end
    assert(isfield(saved_opts.mppca_options, 'center_data'), 'local_preprocess_one:SavedOptions', ...
        'Saved preprocessing_options.mppca_options lacks center_data.');
    assert(logical(saved_opts.enable_mppca) && logical(saved_opts.mppca_options.center_data), ...
        'local_preprocess_one:SavedOptions', 'Saved MAT does not record required MP-PCA settings.');
    assert(~logical(saved_opts.enable_figures) && isempty(saved_opts.max_slices), ...
        'local_preprocess_one:SavedOptions', 'Saved MAT does not record required full-stack/figures-off settings.');
    assert(isscalar(saved.TR) && isfinite(saved.TR) && isscalar(saved.VPS) && isfinite(saved.VPS), ...
        'local_preprocess_one:SavedMetadata', 'Saved TR/VPS are invalid.');
    qc = struct('schema_version', 'local-preprocess-qc-v1.1', ...
        'subject_id', subject_id, 'stack', stack_name, ...
        'input_dicom_dir', input_dicom_dir, 'preprocessed_mat', output_mat_path, ...
        'group_count', size(saved.Mag_crop, 4), 'slice_count', size(saved.Mag_crop, 4), ...
        'weights_per_group', 10, 'mag_crop_shape', size(saved.Mag_crop), ...
        'TR', double(saved.TR), 'VPS', double(saved.VPS), ...
        'mppca_enabled', logical(saved_opts.enable_mppca), ...
        'mppca_center_data', logical(saved.mppca_center_data), ...
        'mind_alpha', double(saved_opts.mind_alpha), ...
        'figures_enabled', logical(saved_opts.enable_figures), 'max_slices', saved_opts.max_slices);
    fid = fopen(qc_path, 'w');
    assert(fid ~= -1, 'local_preprocess_one:QCWrite', 'Cannot create %s', qc_path);
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s\n', jsonencode(qc));
end
