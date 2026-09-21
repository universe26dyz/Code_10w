function build_synthetic_apparent_maps(source_mat_path, synthetic_signal_path, output_mat_path, cache_root)
%BUILD_SYNTHETIC_APPARENT_MAPS Match PSF-reprojected signals with original MultiMap code.
%   Only the signal tensor is synthetic. Protocol metadata and dictionary
%   physics come from the exact current preprocessed.mat and the original
%   function_T1T2_10HB_bssfp implementation.

    arguments
        source_mat_path (1, :) char
        synthetic_signal_path (1, :) char
        output_mat_path (1, :) char
        cache_root (1, :) char = ''
    end
    assert(isfile(source_mat_path), 'build_synthetic_apparent_maps:SourceMissing', ...
        'Source MAT is missing: %s', source_mat_path);
    assert(isfile(synthetic_signal_path), 'build_synthetic_apparent_maps:SignalMissing', ...
        'Synthetic signal MAT is missing: %s', synthetic_signal_path);
    assert(~isfile(output_mat_path), 'build_synthetic_apparent_maps:OutputExists', ...
        'Refusing to overwrite mapping output: %s', output_mat_path);

    script_dir = fileparts(mfilename('fullpath'));
    code_root = fileparts(fileparts(fileparts(fileparts(script_dir))));
    utility_dir = fullfile(code_root, 'trad', 'modules', 'module_01_preprocess_matlab', 'hhz_original', 'Utility');
    matcher_path = fullfile(utility_dir, 'function_T1T2_10HB_bssfp.m');
    assert(isfile(matcher_path), 'build_synthetic_apparent_maps:MatcherMissing', ...
        'Original MultiMap matcher is missing: %s', matcher_path);
    addpath(utility_dir);

    S = load(source_mat_path, 'Acq_time', 'FA', 'TI', 'T2_prep', ...
        'Info_by_slice', 'final_data_semantics');
    required = {'Acq_time', 'FA', 'TI', 'T2_prep', 'Info_by_slice', 'final_data_semantics'};
    for field_index = 1:numel(required)
        assert(isfield(S, required{field_index}), 'build_synthetic_apparent_maps:MissingField', ...
            'Source MAT lacks %s: %s', required{field_index}, source_mat_path);
    end
    assert(strcmp(char(S.final_data_semantics), 'MP-PCA(MIND_mag_reg)'), ...
        'build_synthetic_apparent_maps:Semantics', 'Source final_data_semantics is not MP-PCA(MIND_mag_reg).');
    Y = load(synthetic_signal_path, 'Mag_synthetic', 'group_idx', 'weight_idx');
    assert(isfield(Y, 'Mag_synthetic'), 'build_synthetic_apparent_maps:SignalField', ...
        'Synthetic signal MAT lacks Mag_synthetic.');
    assert(ndims(Y.Mag_synthetic) == 4 && size(Y.Mag_synthetic, 3) == 10, ...
        'build_synthetic_apparent_maps:SignalShape', 'Mag_synthetic must be [row,col,10,group].');
    groups = size(Y.Mag_synthetic, 4);
    assert(isequal(double(Y.group_idx(:))', 0:groups-1), ...
        'build_synthetic_apparent_maps:GroupOrder', 'group_idx must be 0..N-1.');
    assert(isequal(double(Y.weight_idx(:))', 0:9), ...
        'build_synthetic_apparent_maps:WeightOrder', 'weight_idx must be 0..9.');
    assert(size(S.Acq_time, 1) == 10 && size(S.Acq_time, 2) == groups && numel(S.Info_by_slice) == groups, ...
        'build_synthetic_apparent_maps:MetadataShape', 'Current source metadata does not match synthetic groups.');
    assert(isequal(double(S.FA(:))', [45, 45, 45]) && isequal(double(S.TI(:))', [50, 150]) ...
        && isequal(double(S.T2_prep(:))', [35, 45, 55]), ...
        'build_synthetic_apparent_maps:Protocol', 'Current protocol differs from the authoritative HHZ values.');

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

    rows = size(Y.Mag_synthetic, 1); cols = size(Y.Mag_synthetic, 2);
    t1_ms = zeros(groups, rows, cols, 'single'); cache_records=cell(groups,1);
    t2_ms = zeros(groups, rows, cols, 'single');
    valid_mask = false(groups, rows, cols);
    for group_index = 1:groups
        info = S.Info_by_slice{group_index};
        assert(isfield(info, 'RepetitionTime') && isfield(info, 'EchoTrainLength'), ...
            'build_synthetic_apparent_maps:Info', 'Group %d lacks TR/VPS.', group_index - 1);
        [dictionary, cache_record] = get_or_build_multimap_dictionary(info, S.Acq_time(:, group_index), S.FA, Tlist, S.T2_prep, S.TI, cache_root, fullfile(utility_dir,'sim_T1T2_10HB_bssfp.m'));
        [T1map, T2map, ~, valid] = match_multimap_dictionary(Y.Mag_synthetic(:, :, :, group_index), dictionary);
        valid = valid & isfinite(T1map) & isfinite(T2map) & T1map > 0 & T2map > 0; cache_records{group_index}=cache_record;
        t1_ms(group_index, :, :) = reshape(single(T1map), [1, rows, cols]);
        t2_ms(group_index, :, :) = reshape(single(T2map), [1, rows, cols]);
        valid_mask(group_index, :, :) = reshape(valid, [1, rows, cols]);
    end
    assert(mean(~isfinite(t1_ms), 'all') <= 1e-3 && mean(~isfinite(t2_ms), 'all') <= 1e-3, ...
        'build_synthetic_apparent_maps:NonFinite', 'Apparent maps contain an abnormal non-finite fraction.');
    group_idx = int32((0:groups-1)');
    psf_samples = int32(8);
    matcher_function = matcher_path;
    physics_parameters_changed = false;
    cache_hits=sum(cellfun(@(x) strcmp(x.cache_status,'hit'),cache_records)); cache_misses=sum(cellfun(@(x) strcmp(x.cache_status,'miss'),cache_records)); cache_provenance=struct('cache_enabled',~isempty(cache_root),'cache_root',cache_root,'cache_hits',cache_hits,'cache_misses',cache_misses,'unique_signatures',numel(unique(cellfun(@(x)x.signature,cache_records,'UniformOutput',false))),'per_group',{cache_records});
    save(output_mat_path, 't1_ms', 't2_ms', 'valid_mask', 'group_idx', 'psf_samples', 'cache_provenance', ...
        'source_mat_path', 'synthetic_signal_path', 'matcher_function', ...
        'physics_parameters_changed', '-v7');
end
