function preprocess_stack_v1(input_dicom_dir, output_mat_path, opts)
%PREPROCESS_STACK_V1 HHZ-compatible 10-weight preprocessing for one stack.
%   This Phase-1 entry point preserves the HHZ v1 acquisition contract:
%   NumImg=10, FA=[45 45 45], TI=[50 150] ms, T2_prep=[35 45 55] ms.
%   It sorts DICOMs by AcquisitionTime, reshapes them to [Nx Ny 10 Nslice],
%   optionally runs the vendored MP-PCA routine, registers weights 2..10 to
%   weight 1 using the vendored MIND code, and saves MIND data as Mag_crop.
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
    if opts.enable_mppca
        assert(isstruct(opts.mppca_options), 'preprocess_stack_v1:MPPCAOptions', ...
            'mppca_options must be a struct when enable_mppca is true.');
    end

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
    for k = 1:n_files
        source_path = fullfile(dicoms(k).folder, dicoms(k).name);
        info_k = dicominfo(source_path);
        assert(isfield(info_k, 'AcquisitionTime'), ...
            'preprocess_stack_v1:AcquisitionTime', ...
            'DICOM lacks AcquisitionTime: %s', source_path);
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
    end

    [acquisition_times_sorted, order] = sort(acquisition_times_unsorted, 'ascend');
    raw_images = raw_images(:, :, order);
    infos_sorted = infos_unsorted(order);
    source_filenames_sorted = source_filenames_unsorted(order);

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
        infos_sorted = infos_sorted(1:NumImg*n_slices);
    end

    crop_rows = nx/4 : 3*nx/4;
    crop_cols = ny/4 : 3*ny/4;
    Mag_crop_raw = Mag(crop_rows, crop_cols, :, :);
    if opts.enable_mppca
        [Mag_for_mind, MPPCA_sigma_map, MPPCA_rank_map, ...
            MPPCA_support_mask, MPPCA_residual, MPPCA_report] = ...
            mppca_denoise_multimap_stack(Mag_crop_raw, opts.mppca_options);
    else
        Mag_for_mind = Mag_crop_raw;
        MPPCA_sigma_map = [];
        MPPCA_rank_map = [];
        MPPCA_support_mask = [];
        MPPCA_residual = [];
        MPPCA_report = [];
    end

    MIND_mag_reg = zeros(size(Mag_for_mind), 'like', Mag_for_mind);
    MIND_mag_reg(:, :, 1, :) = Mag_for_mind(:, :, 1, :);
    for slice_idx = 1:n_slices
        fixed_image = single(Mag_for_mind(:, :, 1, slice_idx));
        for weight_idx = 2:NumImg
            moving_image = single(Mag_for_mind(:, :, weight_idx, slice_idx));
            [~, ~, registered_image] = deformableReg2Dmind_asym_nodisplay( ...
                fixed_image, moving_image, opts.mind_alpha);
            MIND_mag_reg(:, :, weight_idx, slice_idx) = registered_image;
        end
    end

    % Critical v1 data-flow correction: later Python code must consume MIND data.
    Mag_crop = MIND_mag_reg;
    Info = infos_sorted{1};
    Info_by_slice = cell(1, n_slices);
    HB1_source_file_per_slice = cell(1, n_slices);
    for slice_idx = 1:n_slices
        source_idx = 1 + (slice_idx - 1) * NumImg;
        Info_by_slice{slice_idx} = infos_sorted{source_idx};
        HB1_source_file_per_slice{slice_idx} = source_filenames_sorted{source_idx};
    end
    assert(isfield(Info, 'RepetitionTime') && isfield(Info, 'EchoTrainLength'), ...
        'preprocess_stack_v1:ProtocolMetadata', ...
        'HB1 DICOM must contain RepetitionTime and EchoTrainLength.');
    TR = Info.RepetitionTime;
    VPS = Info.EchoTrainLength;
    FA = [45, 45, 45];
    TI = [50, 150];
    T2_prep = [35, 45, 55];
    protocol_version = 'hhz_v1';
    geometry_rule = 'After MIND, all 10 weights in each group use HB1 geometry.';
    preprocessing_options = opts;
    crop_indices = struct('rows', crop_rows, 'cols', crop_cols);

    if opts.enable_figures
        figure('Name', 'HHZ v1 preprocessing', 'Color', 'w');
        imshow(Mag_crop(:, :, 1, 1), []);
        title('HB1 MIND reference, slice 1');
    end

    output_dir = fileparts(output_mat_path);
    assert(isfolder(output_dir), 'preprocess_stack_v1:OutputDirectory', ...
        'Output directory must already exist: %s', output_dir);
    save(output_mat_path, 'Mag', 'Mag_crop', 'Mag_crop_raw', 'MIND_mag_reg', ...
        'Acq_time', 'TR', 'VPS', 'FA', 'TI', 'T2_prep', 'Info', ...
        'Info_by_slice', 'HB1_source_file_per_slice', 'source_filenames_sorted', ...
        'crop_indices', 'geometry_rule', 'protocol_version', ...
        'preprocessing_options', 'MPPCA_sigma_map', 'MPPCA_rank_map', ...
        'MPPCA_support_mask', 'MPPCA_residual', 'MPPCA_report', '-v7.3');
end
