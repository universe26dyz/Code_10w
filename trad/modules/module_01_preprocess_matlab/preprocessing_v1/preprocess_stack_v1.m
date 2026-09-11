function preprocess_stack_v1(input_dicom_dir, output_mat_path, opts)
%PREPROCESS_STACK_V1 HHZ-compatible 10-weight preprocessing for one stack.
%   This Phase-1 entry point preserves the HHZ v1 acquisition contract:
%   NumImg=10, FA=[45 45 45], TI=[50 150] ms, T2_prep=[35 45 55] ms.
%   It sorts DICOMs by AcquisitionTime, verifies native-slice groups, reshapes
%   them to [Nx Ny 10 Nslice], registers weights 2..10 to weight 1 with MIND,
%   optionally runs the vendored MP-PCA routine on the MIND result, and saves
%   the final Python input as Mag_crop.
%
%   Required opts fields (no inferred acquisition defaults):
%     max_slices       [] for every slice, or a positive integer for smoke
%     enable_figures   logical scalar
%     enable_mppca     logical scalar
%     mind_alpha       positive finite scalar
%     mppca_options    struct (required when enable_mppca is true)

    arguments
        input_dicom_dir (1, :) char
        output_mat_path (1, :) char
        opts (1, 1) struct
    end

    required_fields = {'max_slices', 'enable_figures', 'enable_mppca', ...
        'mind_alpha', 'mppca_options'};
    for k = 1:numel(required_fields)
        assert(isfield(opts, required_fields{k}), ...
            'preprocess_stack_v1:MissingOption', ...
            'opts.%s is required.', required_fields{k});
    end
    assert(isfolder(input_dicom_dir), 'preprocess_stack_v1:InputDirectory', ...
        'DICOM directory does not exist: %s', input_dicom_dir);
    assert(~isfile(output_mat_path), 'preprocess_stack_v1:OutputExists', ...
        'Refusing to overwrite existing output MAT: %s', output_mat_path);
    assert(islogical(opts.enable_figures) && isscalar(opts.enable_figures), ...
        'preprocess_stack_v1:FigureOption', 'enable_figures must be logical scalar.');
    assert(islogical(opts.enable_mppca) && isscalar(opts.enable_mppca), ...
        'preprocess_stack_v1:MPPCAOption', 'enable_mppca must be logical scalar.');
    assert(isnumeric(opts.mind_alpha) && isscalar(opts.mind_alpha) && ...
        isfinite(opts.mind_alpha) && opts.mind_alpha > 0, ...
        'preprocess_stack_v1:MINDAlpha', 'mind_alpha must be positive and finite.');
    if isempty(opts.max_slices)
        max_slices = [];
    else
        assert(isnumeric(opts.max_slices) && isscalar(opts.max_slices) && ...
            isfinite(opts.max_slices) && opts.max_slices >= 1 && ...
            opts.max_slices == floor(opts.max_slices), ...
            'preprocess_stack_v1:MaxSlices', ...
            'max_slices must be [] or a positive integer.');
        max_slices = opts.max_slices;
    end
    assert(isstruct(opts.mppca_options), 'preprocess_stack_v1:MPPCAOptions', ...
        'mppca_options must be an explicit struct.');
    required_mppca_fields = {'patch_size', 'center_data', 'clip_negative', ...
        'use_parallel', 'mask_threshold_fraction', 'min_component_size', ...
        'fill_holes', 'save_residual', 'overwrite'};
    for k = 1:numel(required_mppca_fields)
        assert(isfield(opts.mppca_options, required_mppca_fields{k}), ...
            'preprocess_stack_v1:MPPCAOptions', ...
            'opts.mppca_options.%s must be set explicitly.', required_mppca_fields{k});
    end
    assert(islogical(opts.mppca_options.center_data) && ...
        isscalar(opts.mppca_options.center_data), ...
        'preprocess_stack_v1:MPPCACenterData', ...
        'mppca_options.center_data must be an explicit logical scalar.');

    this_dir = fileparts(mfilename('fullpath'));
    addpath(this_dir);
    dicoms = dir(fullfile(input_dicom_dir, '*.dcm'));
    assert(~isempty(dicoms), 'preprocess_stack_v1:NoDicom', ...
        'No .dcm files found in %s', input_dicom_dir);

    n_files = numel(dicoms);
    raw_images = [];
    acquisition_times_unsorted = zeros(1, n_files);
    infos_unsorted = cell(1, n_files);
    source_filenames_unsorted = cell(1, n_files);
    source_basenames_unsorted = cell(1, n_files);
    sop_instance_uids_unsorted = cell(1, n_files);
    series_instance_uids_unsorted = cell(1, n_files);
    for k = 1:n_files
        source_path = fullfile(dicoms(k).folder, dicoms(k).name);
        info_k = dicominfo(source_path);
        assert(isfield(info_k, 'AcquisitionTime'), ...
            'preprocess_stack_v1:AcquisitionTime', ...
            'DICOM lacks AcquisitionTime: %s', source_path);
        assert(isfield(info_k, 'SOPInstanceUID') && ~isempty(info_k.SOPInstanceUID), ...
            'preprocess_stack_v1:SOPInstanceUID', ...
            'DICOM lacks SOPInstanceUID: %s', source_path);
        assert(isfield(info_k, 'SeriesInstanceUID') && ~isempty(info_k.SeriesInstanceUID), ...
            'preprocess_stack_v1:SeriesInstanceUID', ...
            'DICOM lacks SeriesInstanceUID: %s', source_path);
        image_k = double(dicomread(source_path));
        if k == 1
            raw_images = zeros([size(image_k), n_files]);
        else
            assert(isequal(size(image_k), size(raw_images(:, :, 1))), ...
                'preprocess_stack_v1:ImageSize', ...
                'DICOM image size differs from first file: %s', source_path);
        end
        raw_images(:, :, k) = image_k;
        acquisition_times_unsorted(k) = HHMMSS2Sec(info_k.AcquisitionTime) * 1e3;
        infos_unsorted{k} = info_k;
        source_filenames_unsorted{k} = source_path;
        source_basenames_unsorted{k} = dicoms(k).name;
        sop_instance_uids_unsorted{k} = char(info_k.SOPInstanceUID);
        series_instance_uids_unsorted{k} = char(info_k.SeriesInstanceUID);
    end

    [acquisition_times_sorted, order] = sort(acquisition_times_unsorted, 'ascend');
    raw_images = raw_images(:, :, order);
    infos_sorted = infos_unsorted(order);
    source_filenames_sorted = source_filenames_unsorted(order);
    source_basenames_sorted = source_basenames_unsorted(order);
    sop_instance_uids_sorted = sop_instance_uids_unsorted(order);
    series_instance_uids_sorted = series_instance_uids_unsorted(order);

    NumImg = 10;
    assert(mod(n_files, NumImg) == 0, 'preprocess_stack_v1:IncompleteGroups', ...
        'Expected a multiple of %d DICOMs, received %d.', NumImg, n_files);
    [nx, ny, ~] = size(raw_images);
    assert(mod(nx, 4) == 0 && mod(ny, 4) == 0, ...
        'preprocess_stack_v1:HHZCrop', ...
        'HHZ crop requires Nx and Ny divisible by four; got [%d %d].', nx, ny);
    n_slices_available = n_files / NumImg;
    Mag = reshape(raw_images, [nx, ny, NumImg, n_slices_available]);
    Acq_time = reshape(acquisition_times_sorted, [NumImg, n_slices_available]);

    if isempty(max_slices)
        n_slices = n_slices_available;
    else
        assert(max_slices <= n_slices_available, ...
            'preprocess_stack_v1:MaxSlicesRange', ...
            'max_slices=%d exceeds %d available slices.', max_slices, n_slices_available);
        n_slices = max_slices;
        Mag = Mag(:, :, :, 1:n_slices);
        Acq_time = Acq_time(:, 1:n_slices);
        source_filenames_sorted = source_filenames_sorted(1:NumImg*n_slices);
        source_basenames_sorted = source_basenames_sorted(1:NumImg*n_slices);
        infos_sorted = infos_sorted(1:NumImg*n_slices);
        sop_instance_uids_sorted = sop_instance_uids_sorted(1:NumImg*n_slices);
        series_instance_uids_sorted = series_instance_uids_sorted(1:NumImg*n_slices);
    end

    validate_constant_protocol(infos_sorted, source_filenames_sorted);
    validate_group_geometry(infos_sorted, source_filenames_sorted, NumImg);
    assert(numel(unique(series_instance_uids_sorted)) == 1, ...
        'preprocess_stack_v1:SeriesInstanceUID', ...
        'Output DICOMs must share one SeriesInstanceUID; found: %s', ...
        strjoin(unique(series_instance_uids_sorted), ', '));

    crop_rows = nx/4 : 3*nx/4;
    crop_cols = ny/4 : 3*ny/4;
    Mag_crop_unregistered = Mag(crop_rows, crop_cols, :, :);

    MIND_mag_reg = zeros(size(Mag_crop_unregistered), 'like', Mag_crop_unregistered);
    MIND_mag_reg(:, :, 1, :) = Mag_crop_unregistered(:, :, 1, :);
    for slice_idx = 1:n_slices
        fixed_image = single(Mag_crop_unregistered(:, :, 1, slice_idx));
        for weight_idx = 2:NumImg
            moving_image = single(Mag_crop_unregistered(:, :, weight_idx, slice_idx));
            [~, ~, registered_image] = deformableReg2Dmind_asym_nodisplay( ...
                fixed_image, moving_image, opts.mind_alpha);
            MIND_mag_reg(:, :, weight_idx, slice_idx) = registered_image;
        end
    end

    if opts.enable_mppca
        [Mag_crop, MPPCA_sigma_map, MPPCA_rank_map, MPPCA_support_mask, ...
            MPPCA_residual, MPPCA_report] = ...
            mppca_denoise_multimap_stack(MIND_mag_reg, opts.mppca_options);
        final_data_semantics = 'MP-PCA(MIND_mag_reg)';
    else
        Mag_crop = MIND_mag_reg;
        MPPCA_sigma_map = [];
        MPPCA_rank_map = [];
        MPPCA_support_mask = [];
        MPPCA_residual = [];
        MPPCA_report = [];
        final_data_semantics = 'MIND_mag_reg';
    end

    % Critical v1 data-flow contract: Python always consumes final Mag_crop.
    Info = infos_sorted{1};
    Info_by_slice = cell(1, n_slices);
    HB1_source_file_per_slice = cell(1, n_slices);
    for slice_idx = 1:n_slices
        source_idx = 1 + (slice_idx - 1) * NumImg;
        Info_by_slice{slice_idx} = infos_sorted{source_idx};
        HB1_source_file_per_slice{slice_idx} = source_filenames_sorted{source_idx};
    end
    TR = Info.RepetitionTime;
    VPS = Info.EchoTrainLength;
    FA = [45, 45, 45];
    TI = [50, 150];
    T2_prep = [35, 45, 55];
    protocol_version = 'hhz_v1';
    geometry_rule = 'After MIND, all 10 weights in each group use HB1 geometry.';
    preprocessing_options = opts;
    crop_indices = struct('rows', crop_rows, 'cols', crop_cols);
    crop_row_start_zero_based = crop_rows(1) - 1;
    crop_col_start_zero_based = crop_cols(1) - 1;
    crop_size_rows_cols = [numel(crop_rows), numel(crop_cols)];
    mppca_center_data = opts.mppca_options.center_data;
    sop_instance_uid_ascii = encode_ascii_matrix(sop_instance_uids_sorted);
    sop_instance_uid_lengths = cellfun(@strlength, string(sop_instance_uids_sorted));
    series_instance_uid_ascii = encode_ascii_matrix(series_instance_uids_sorted);
    series_instance_uid_lengths = cellfun(@strlength, string(series_instance_uids_sorted));
    source_basename_ascii = encode_ascii_matrix(source_basenames_sorted);
    source_basename_lengths = cellfun(@strlength, string(source_basenames_sorted));
    HB1_sop_instance_uid_ascii = sop_instance_uid_ascii(:, 1:NumImg:end);
    HB1_sop_instance_uid_lengths = sop_instance_uid_lengths(1:NumImg:end);

    if opts.enable_figures
        figure('Name', 'HHZ v1 preprocessing', 'Color', 'w');
        imshow(Mag_crop(:, :, 1, 1), []);
        title('HB1 MIND reference, slice 1');
    end

    output_dir = fileparts(output_mat_path);
    assert(isfolder(output_dir), 'preprocess_stack_v1:OutputDirectory', ...
        'Output directory must already exist: %s', output_dir);
    save(output_mat_path, 'Mag', 'Mag_crop', 'Mag_crop_unregistered', 'MIND_mag_reg', ...
        'Acq_time', 'TR', 'VPS', 'FA', 'TI', 'T2_prep', 'Info', ...
        'Info_by_slice', 'HB1_source_file_per_slice', 'source_filenames_sorted', ...
        'crop_indices', 'crop_row_start_zero_based', 'crop_col_start_zero_based', ...
        'crop_size_rows_cols', 'geometry_rule', 'protocol_version', ...
        'preprocessing_options', 'MPPCA_sigma_map', 'MPPCA_rank_map', ...
        'MPPCA_support_mask', 'MPPCA_residual', 'MPPCA_report', ...
        'mppca_center_data', 'final_data_semantics', ...
        'sop_instance_uid_ascii', 'sop_instance_uid_lengths', ...
        'series_instance_uid_ascii', 'series_instance_uid_lengths', ...
        'source_basename_ascii', 'source_basename_lengths', ...
        'HB1_sop_instance_uid_ascii', 'HB1_sop_instance_uid_lengths', '-v7.3');
end

function validate_constant_protocol(infos, source_paths)
    field_names = {'RepetitionTime', 'EchoTrainLength'};
    for field_idx = 1:numel(field_names)
        field_name = field_names{field_idx};
        values = zeros(1, numel(infos));
        for k = 1:numel(infos)
            assert(isfield(infos{k}, field_name) && ~isempty(infos{k}.(field_name)), ...
                'preprocess_stack_v1:ProtocolMetadata', ...
                'DICOM lacks %s: %s', field_name, source_paths{k});
            values(k) = double(infos{k}.(field_name));
        end
        tolerance = 1e-6 * max(1, max(abs(values)));
        assert(max(abs(values - values(1))) <= tolerance, ...
            'preprocess_stack_v1:ProtocolInconsistent', ...
            '%s must be constant within the output stack. Values: %s', ...
            field_name, mat2str(values, 12));
    end
end

function validate_group_geometry(infos, source_paths, num_img)
    assert(mod(numel(infos), num_img) == 0, ...
        'preprocess_stack_v1:IncompleteGroups', 'DICOM count does not form full groups.');
    for group_idx = 1:(numel(infos) / num_img)
        indices = (group_idx - 1) * num_img + (1:num_img);
        reference = infos{indices(1)};
        compare_geometry_field(reference, infos(indices), source_paths(indices), ...
            'ImagePositionPatient', group_idx, false);
        compare_geometry_field(reference, infos(indices), source_paths(indices), ...
            'ImageOrientationPatient', group_idx, false);
        compare_geometry_field(reference, infos(indices), source_paths(indices), ...
            'PixelSpacing', group_idx, false);
        compare_geometry_field(reference, infos(indices), source_paths(indices), ...
            'Rows', group_idx, true);
        compare_geometry_field(reference, infos(indices), source_paths(indices), ...
            'Columns', group_idx, true);
        reference_has_thickness = isfield(reference, 'SliceThickness') && ...
            ~isempty(reference.SliceThickness);
        for k = 1:num_img
            current_has_thickness = isfield(infos{indices(k)}, 'SliceThickness') && ...
                ~isempty(infos{indices(k)}.SliceThickness);
            assert(current_has_thickness == reference_has_thickness, ...
                'preprocess_stack_v1:GroupGeometry', ...
                'Group %d has inconsistent SliceThickness presence; file: %s', ...
                group_idx, source_paths{indices(k)});
        end
        if reference_has_thickness
            compare_geometry_field(reference, infos(indices), source_paths(indices), ...
                'SliceThickness', group_idx, false);
        end
    end
end

function compare_geometry_field(reference, group_infos, group_paths, field_name, group_idx, exact)
    assert(isfield(reference, field_name) && ~isempty(reference.(field_name)), ...
        'preprocess_stack_v1:GroupGeometry', ...
        'Group %d HB1 lacks %s: %s', group_idx, field_name, group_paths{1});
    reference_value = double(reference.(field_name));
    values_text = cell(1, numel(group_infos));
    for k = 1:numel(group_infos)
        assert(isfield(group_infos{k}, field_name) && ~isempty(group_infos{k}.(field_name)), ...
            'preprocess_stack_v1:GroupGeometry', ...
            'Group %d weight %d lacks %s: %s', group_idx, k - 1, field_name, group_paths{k});
        value = double(group_infos{k}.(field_name));
        values_text{k} = mat2str(value, 12);
        if exact
            equal = isequal(value, reference_value);
        else
            equal = isequal(size(value), size(reference_value)) && ...
                all(abs(value(:) - reference_value(:)) <= ...
                1e-6 * max(1, max(abs(reference_value(:)))));
        end
        assert(equal, 'preprocess_stack_v1:GroupGeometry', ...
            'Group %d %s differs from HB1. Values: %s', group_idx, field_name, ...
            strjoin(values_text(1:k), ' | '));
    end
end

function ascii_matrix = encode_ascii_matrix(values)
    values = cellfun(@char, values, 'UniformOutput', false);
    max_length = max(cellfun(@numel, values));
    ascii_matrix = zeros(max_length, numel(values), 'uint8');
    for k = 1:numel(values)
        ascii_matrix(1:numel(values{k}), k) = uint8(values{k});
    end
end
