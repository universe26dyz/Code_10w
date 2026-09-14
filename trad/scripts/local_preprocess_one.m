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
    qc = struct('schema_version', 'local-preprocess-qc-v1', ...
        'subject_id', subject_id, 'stack', stack_name, ...
        'input_dicom_dir', input_dicom_dir, 'preprocessed_mat', output_mat_path, ...
        'mppca_enabled', true, 'figures_enabled', false, 'max_slices', []);
    fid = fopen(qc_path, 'w');
    assert(fid ~= -1, 'local_preprocess_one:QCWrite', 'Cannot create %s', qc_path);
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s\n', jsonencode(qc));
end
