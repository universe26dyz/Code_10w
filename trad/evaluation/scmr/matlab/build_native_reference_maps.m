function build_native_reference_maps(source_mat_path, output_mat_path)
%BUILD_NATIVE_REFERENCE_MAPS Run only the original MultiMap dictionary matcher.
%   The source is an already-complete Code_10w preprocessed.mat.  This wrapper
%   does not import DICOM, crop, MIND-register, or run MP-PCA; it reads final
%   Mag_crop and the protocol fields that were saved with that exact input.

    arguments
        source_mat_path (1, :) char
        output_mat_path (1, :) char
    end
    assert(isfile(source_mat_path), 'build_native_reference_maps:SourceMissing', ...
        'Source MAT is missing: %s', source_mat_path);
    assert(~isfile(output_mat_path), 'build_native_reference_maps:OutputExists', ...
        'Refusing to overwrite mapping output: %s', output_mat_path);
    output_dir = fileparts(output_mat_path);
    assert(isfolder(output_dir), 'build_native_reference_maps:OutputDirectory', ...
        'Output directory must already exist: %s', output_dir);

    script_dir = fileparts(mfilename('fullpath'));
    code_root = fileparts(fileparts(fileparts(fileparts(script_dir))));
    utility_dir = fullfile(code_root, 'trad', 'modules', 'module_01_preprocess_matlab', 'hhz_original', 'Utility');
    assert(isfile(fullfile(utility_dir, 'function_T1T2_10HB_bssfp.m')), ...
        'build_native_reference_maps:OriginalMatcherMissing', 'Original MultiMap matcher is missing.');
    addpath(utility_dir);

    S = load(source_mat_path, 'Mag_crop', 'Acq_time', 'FA', 'TI', 'T2_prep', ...
        'Info', 'Info_by_slice', 'final_data_semantics');
    fields = {'Mag_crop', 'Acq_time', 'FA', 'TI', 'T2_prep', 'Info', 'Info_by_slice', 'final_data_semantics'};
    for field_index = 1:numel(fields)
        assert(isfield(S, fields{field_index}), 'build_native_reference_maps:MissingField', ...
            'Source MAT lacks %s: %s', fields{field_index}, source_mat_path);
    end
    assert(strcmp(char(S.final_data_semantics), 'MP-PCA(MIND_mag_reg)'), ...
        'build_native_reference_maps:Semantics', 'Source final_data_semantics is not MP-PCA(MIND_mag_reg).');
    % MATLAB elides a trailing singleton dimension on load, so one-group
    % inputs arrive as [row,col,10] even though their logical layout is
    % [row,col,10,1].  size(...,4) remains 1 in that case.
    assert(size(S.Mag_crop, 3) == 10, ...
        'build_native_reference_maps:Shape', 'Mag_crop must be [row,col,10,group].');
    assert(size(S.Acq_time, 1) == 10 && size(S.Acq_time, 2) == size(S.Mag_crop, 4), ...
        'build_native_reference_maps:TimingShape', 'Acq_time must be [10,group].');
    assert(isequal(double(S.FA(:))', [45, 45, 45]), 'build_native_reference_maps:Protocol', ...
        'Current source FA differs from the authoritative HHZ [45,45,45] degrees.');
    assert(isequal(double(S.TI(:))', [50, 150]), 'build_native_reference_maps:Protocol', ...
        'Current source TI differs from the authoritative HHZ [50,150] ms.');
    assert(isequal(double(S.T2_prep(:))', [35, 45, 55]), 'build_native_reference_maps:Protocol', ...
        'Current source T2_prep differs from the authoritative HHZ [35,45,55] ms.');

    T1_list = [20:20:500, 505:5:1500, 1520:20:2500];
    T2_list = [5:5:100, 110:10:200];
    B1_list = 0.1:0.05:1.2;
    Tlist = zeros(numel(T1_list) * numel(T2_list) * numel(B1_list), 5);
    index = 1;
    for t1_index = 1:numel(T1_list)
        for t2_index = 1:numel(T2_list)
            for b1_index = 1:numel(B1_list)
                if T1_list(t1_index) > T2_list(t2_index)
                    Tlist(index, 1:3) = [T1_list(t1_index), T2_list(t2_index), B1_list(b1_index)];
                    index = index + 1;
                end
            end
        end
    end
    Tlist(all(Tlist == 0, 2), :) = [];
    Tlist = unique(Tlist, 'rows');

    rows = size(S.Mag_crop, 1); cols = size(S.Mag_crop, 2); groups = size(S.Mag_crop, 4);
    t1_ms = zeros(groups, rows, cols, 'single');
    t2_ms = zeros(groups, rows, cols, 'single');
    valid_mask = false(groups, rows, cols);
    for group_index = 1:groups
        info = S.Info_by_slice{group_index};
        assert(isfield(info, 'RepetitionTime') && isfield(info, 'EchoTrainLength'), ...
            'build_native_reference_maps:Info', 'Group %d lacks RepetitionTime/EchoTrainLength.', group_index - 1);
        [T1map, T2map] = function_T1T2_10HB_bssfp( ...
            S.Mag_crop(:, :, :, group_index), info, S.Acq_time(:, group_index), ...
            S.FA, Tlist, S.T2_prep, S.TI);
        assert(isequal(size(T1map), [rows, cols]) && isequal(size(T2map), [rows, cols]), ...
            'build_native_reference_maps:MapShape', 'Matcher output shape is invalid for group %d.', group_index - 1);
        valid = isfinite(T1map) & isfinite(T2map) & T1map > 0 & T2map > 0;
        t1_ms(group_index, :, :) = reshape(single(T1map), [1, rows, cols]);
        t2_ms(group_index, :, :) = reshape(single(T2map), [1, rows, cols]);
        valid_mask(group_index, :, :) = reshape(valid, [1, rows, cols]);
    end
    assert(mean(~isfinite(t1_ms), 'all') <= 1e-3 && mean(~isfinite(t2_ms), 'all') <= 1e-3, ...
        'build_native_reference_maps:NonFinite', 'T1/T2 maps contain an abnormal non-finite fraction.');
    group_idx = int32((0:groups-1)');
    source_final_data_semantics = char(S.final_data_semantics);
    source_protocol = struct('FA_deg', double(S.FA(:))', 'TI_ms', double(S.TI(:))', ...
        'T2_prep_ms', double(S.T2_prep(:))');
    save(output_mat_path, 't1_ms', 't2_ms', 'valid_mask', 'group_idx', ...
        'source_mat_path', 'source_final_data_semantics', 'source_protocol', '-v7');
end
