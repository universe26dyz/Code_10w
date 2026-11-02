% Run_MPPCA_All_Stacks.m
% 对已经完成多权重配准的 2ch / 4ch / sax 数据执行逐切片 patch MP-PCA。
%
% 前置条件：
%   1) PreData.m 已经执行 Mag_crop = MIND_mag_reg;
%   2) 当前目录包含：
%        2ch_data_all.mat
%        4ch_data_all.mat
%        sax_data_all.mat
%
% 输出：
%        2ch_data_all_mppca.mat
%        4ch_data_all_mppca.mat
%        sax_data_all_mppca.mat
%
% 为了无缝衔接原 mapping 程序，输出文件中的 Mag_crop 会被替换为
% 去噪后的数据；配准后但未去噪的数据另存为 Mag_crop_raw。
%
% 参考：
%   Veraart et al., NeuroImage, 2016, 142:394-406.
%   Does et al., Magn Reson Med, 2019, 81:3503-3514.

close all; clc;

%% ======================== 用户配置 ========================
input_files = {
    '2ch_data_all.mat'
    '4ch_data_all.mat'
    'sax_data_all.mat'
};

output_files = {
    '2ch_data_all_mppca_7c.mat'
    '4ch_data_all_mppca_7c.mat'
    'sax_data_all_mppca_7c.mat'
};

% MP-PCA 配置
opts.patch_size = [7, 7];       % 2D 空间 patch；必须是奇数
opts.center_data = true;       % v1 明确保留逐空间样本 measurement-mean centering。
opts.clip_negative = true;      % magnitude 图像输出限制为非负
opts.use_parallel = true;       % 有 Parallel Computing Toolbox 时逐切片 parfor

% 支持区域配置。这里与原 mapping mask 的阈值形式保持一致：
% threshold = mean(mean_image(:)) * mask_threshold_fraction
opts.mask_threshold_fraction = 1/5;
opts.min_component_size = 20;   % 去除小于该像素数的前景连通域
opts.fill_holes = false;

% 输出配置
opts.save_residual = true;      % 保存 raw - denoised；便于检查是否去除了结构
opts.overwrite = false;         % false：已有输出文件时跳过，防止误覆盖
%% =========================================================

assert(numel(input_files) == numel(output_files), ...
    'input_files 与 output_files 数量不一致。');

for f_idx = 1:numel(input_files)
    input_file = input_files{f_idx};
    output_file = output_files{f_idx};

    fprintf('\n============================================================\n');
    fprintf('MP-PCA: %s -> %s\n', input_file, output_file);
    fprintf('============================================================\n');

    if ~exist(input_file, 'file')
        warning('找不到输入文件：%s，已跳过。', input_file);
        continue;
    end

    if exist(output_file, 'file') && ~opts.overwrite
        warning('输出文件已存在且 overwrite=false：%s，已跳过。', output_file);
        continue;
    end

    S = load(input_file);

    required_fields = {'Mag_crop', 'Acq_time', 'Info', 'T2_prep', 'FA', 'TI'};
    for k = 1:numel(required_fields)
        assert(isfield(S, required_fields{k}), ...
            '文件 %s 缺少变量 %s。', input_file, required_fields{k});
    end

    assert(ndims(S.Mag_crop) == 4, ...
        'Mag_crop 应为 [Nx, Ny, 10, Nslice]，当前尺寸为 %s。', ...
        mat2str(size(S.Mag_crop)));
    assert(size(S.Mag_crop, 3) == 10, ...
        '第三维必须为 10 张权重图像，当前为 %d。', size(S.Mag_crop, 3));

    fprintf('输入尺寸：%s\n', mat2str(size(S.Mag_crop)));
    fprintf('patch：%dx%d；center_data=%d；parallel=%d\n', ...
        opts.patch_size(1), opts.patch_size(2), ...
        opts.center_data, opts.use_parallel);

    tic;
    [Mag_mppca, sigma_map, rank_map, support_mask, residual, report] = ...
        mppca_denoise_multimap_stack(S.Mag_crop, opts);
    elapsed = toc;

    % 保留配准后、去噪前的数据；让 Mag_crop 指向去噪结果，便于后续脚本直接使用。
    S.Mag_crop_raw = S.Mag_crop;
    S.Mag_crop = Mag_mppca;
    S.MPPCA_sigma_map = sigma_map;
    S.MPPCA_rank_map = rank_map;
    S.MPPCA_support_mask = support_mask;
    S.MPPCA_options = opts;
    S.MPPCA_report = report;
    S.MPPCA_elapsed_seconds = elapsed;

    if opts.save_residual
        S.MPPCA_residual = residual;
    end

    save(output_file, '-struct', 'S', '-v7.3');

    fprintf('完成并保存：%s\n', output_file);
    fprintf('耗时：%.1f s\n', elapsed);

    ranks = double(rank_map(rank_map > 0));
    sigmas = double(sigma_map(sigma_map > 0 & isfinite(sigma_map)));
    if ~isempty(ranks)
        fprintf('有效区域 rank：median=%.2f, mean=%.2f, range=[%d, %d]\n', ...
            median(ranks), mean(ranks), min(ranks), max(ranks));
    end
    if ~isempty(sigmas)
        fprintf('局部 sigma：median=%.4g, mean=%.4g\n', ...
            median(sigmas), mean(sigmas));
    end
end

fprintf('\n所有可用 stack 的 MP-PCA 处理完成。\n');
