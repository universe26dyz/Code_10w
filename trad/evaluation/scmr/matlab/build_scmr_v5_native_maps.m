function build_scmr_v5_native_maps(source_mat_path, output_mat_path)
%BUILD_SCMR_V5_NATIVE_MAPS Match current cropped observations with SCMR-v5.
%   This deliberately overrides the current HHZ metadata TI=[50,150] with
%   the experimentally reproduced SCMR-v5 dictionary contract TI=[10,100].
%   The input pixels are never resampled or otherwise changed.

    arguments
        source_mat_path (1, :) char
        output_mat_path (1, :) char
    end
    assert(isfile(source_mat_path), 'build_scmr_v5_native_maps:SourceMissing', ...
        'Source MAT is missing: %s', source_mat_path);
    assert(~isfile(output_mat_path), 'build_scmr_v5_native_maps:OutputExists', ...
        'Refusing to overwrite mapping output: %s', output_mat_path);
    output_dir = fileparts(output_mat_path);
    assert(isfolder(output_dir), 'build_scmr_v5_native_maps:OutputDirectory', ...
        'Output directory must already exist: %s', output_dir);

    script_dir = fileparts(mfilename('fullpath'));
    % Keep this production route self-contained.  The previous implementation
    % inferred a sibling MultiMapCode checkout from the Code_10w path.  On the
    % server that sibling was absent, so MATLAB either inherited an unrelated
    % old MultiMap path or failed the matcher preflight before fitting.
    method_dir = fullfile(script_dir, 'scmr_v5_matcher');
    assert(isfile(fullfile(method_dir, 'function_T1T2_10HB_bssfp_qc.m')), ...
        'build_scmr_v5_native_maps:MatcherMissing', 'SCMR-v5 matcher is missing.');
    simulator_dir = fullfile(method_dir, 'matlab', 'dictionary');
    assert(isfile(fullfile(simulator_dir, 'sim_T1T2_10HB_bssfp.m')), ...
        'build_scmr_v5_native_maps:SimulatorMissing', 'SCMR-v5 dictionary simulator is missing.');
    utility_dir = fullfile(method_dir, 'matlab', 'utils');
    assert(isfile(fullfile(utility_dir, 'd2r.m')), ...
        'build_scmr_v5_native_maps:UtilityMissing', 'SCMR-v5 dictionary utility is missing.');
    addpath(method_dir);
    addpath(simulator_dir);
    addpath(utility_dir);
    fprintf('SCMR-v5 matcher root: %s\n', method_dir);
    fprintf('SCMR-v5 matcher: %s\n', which('function_T1T2_10HB_bssfp_qc'));
    fprintf('SCMR-v5 simulator: %s\n', which('sim_T1T2_10HB_bssfp'));

    S = load(source_mat_path, 'Mag_crop', 'Acq_time', 'FA', 'TI', 'T2_prep', ...
        'Info_by_slice', 'final_data_semantics');
    fields = {'Mag_crop', 'Acq_time', 'FA', 'TI', 'T2_prep', 'Info_by_slice', 'final_data_semantics'};
    for field_index = 1:numel(fields)
        assert(isfield(S, fields{field_index}), 'build_scmr_v5_native_maps:MissingField', ...
            'Source MAT lacks %s: %s', fields{field_index}, source_mat_path);
    end
    assert(strcmp(char(S.final_data_semantics), 'MP-PCA(MIND_mag_reg)'), ...
        'build_scmr_v5_native_maps:Semantics', 'Source final_data_semantics is not MP-PCA(MIND_mag_reg).');
    assert(size(S.Mag_crop, 3) == 10 && size(S.Acq_time, 1) == 10, ...
        'build_scmr_v5_native_maps:Shape', 'Expected Mag_crop [row,col,10,group] and Acq_time [10,group].');
    assert(isequal(double(S.FA(:))', [45, 45, 45]), ...
        'build_scmr_v5_native_maps:SourceProtocol', 'Current cropped source FA must be [45,45,45].');
    assert(isequal(double(S.TI(:))', [50, 150]), ...
        'build_scmr_v5_native_maps:SourceProtocol', 'Current cropped source TI must be [50,150].');
    assert(isequal(double(S.T2_prep(:))', [35, 45, 55]), ...
        'build_scmr_v5_native_maps:SourceProtocol', 'Current cropped source T2prep must be [35,45,55].');

    % Exact production grid and SCMR-v5 protocol proven against retained HHZ maps.
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
    fit_fa_deg = [45, 45, 45];
    fit_ti_ms = [10, 100];
    fit_t2prep_ms = [35, 45, 55];

    rows = size(S.Mag_crop, 1); cols = size(S.Mag_crop, 2); groups = size(S.Mag_crop, 4);
    t1_ms = nan(groups, rows, cols, 'single');
    t2_ms = nan(groups, rows, cols, 'single');
    b1 = nan(groups, rows, cols, 'single');
    valid_mask = false(groups, rows, cols);
    match_score = nan(groups, rows, cols, 'single');
    fitting_nrmse = nan(groups, rows, cols, 'single');
    for group_index = 1:groups
        info = S.Info_by_slice{group_index};
        assert(isfield(info, 'RepetitionTime') && isfield(info, 'EchoTrainLength'), ...
            'build_scmr_v5_native_maps:Info', 'Group %d lacks RepetitionTime/EchoTrainLength.', group_index - 1);
        [T1map, T2map, B1map, qc] = function_T1T2_10HB_bssfp_qc( ...
            S.Mag_crop(:, :, :, group_index), info, S.Acq_time(:, group_index), ...
            fit_fa_deg, Tlist, fit_t2prep_ms, fit_ti_ms);
        valid = qc.valid_match_map & isfinite(T1map) & isfinite(T2map) & isfinite(B1map) ...
            & T1map > 0 & T2map > 0 & B1map > 0;
        T1map(~valid) = NaN; T2map(~valid) = NaN; B1map(~valid) = NaN;
        t1_ms(group_index, :, :) = reshape(single(T1map), [1, rows, cols]);
        t2_ms(group_index, :, :) = reshape(single(T2map), [1, rows, cols]);
        b1(group_index, :, :) = reshape(single(B1map), [1, rows, cols]);
        valid_mask(group_index, :, :) = reshape(valid, [1, rows, cols]);
        match_score(group_index, :, :) = reshape(qc.match_score_map, [1, rows, cols]);
        fitting_nrmse(group_index, :, :) = reshape(qc.fitting_nrmse_map, [1, rows, cols]);
    end
    assert(any(valid_mask, 'all'), 'build_scmr_v5_native_maps:Empty', 'No valid SCMR-v5 pixels were matched.');
    group_idx = int32((0:groups-1)');
    source_protocol = struct('FA_deg', double(S.FA(:))', 'TI_ms', double(S.TI(:))', ...
        'T2_prep_ms', double(S.T2_prep(:))');
    fit_protocol = struct('FA_deg', fit_fa_deg, 'TI_ms', fit_ti_ms, ...
        'T2_prep_ms', fit_t2prep_ms, 'timing_formula', ...
        'delta_AcquisitionTime-TR*(EchoTrainLength+10)-preparation_delay', ...
        'normalization', 'L2 normalized absolute 10D inner product', ...
        'refinement', false, 'qc_score_threshold', 0.90);
    source_final_data_semantics = char(S.final_data_semantics);
    save(output_mat_path, 't1_ms', 't2_ms', 'b1', 'valid_mask', 'match_score', ...
        'fitting_nrmse', 'group_idx', 'source_mat_path', 'source_protocol', ...
        'fit_protocol', 'source_final_data_semantics', '-v7');
end
