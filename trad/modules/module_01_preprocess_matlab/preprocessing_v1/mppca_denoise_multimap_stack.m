function [Mag_denoised, sigma_map, rank_map, support_mask, residual, report] = ...
    mppca_denoise_multimap_stack(Mag_registered, opts)
%MPPCA_DENOISE_MULTIMAP_STACK 对 10-heartbeat mapping 数据做逐切片 patch MP-PCA。
%
% 输入
%   Mag_registered : [Nx, Ny, 10, Nslice]
%                    已经完成同切片内多权重配准的数据。
%   opts            : 配置结构体，字段见 apply_default_options。
%
% 输出
%   Mag_denoised : 与输入同尺寸、同数据类型的去噪数据
%   sigma_map    : [Nx, Ny, Nslice]，局部噪声标准差估计
%   rank_map     : [Nx, Ny, Nslice]，MP 判据保留的主成分数
%   support_mask : [Nx, Ny, Nslice]，实际处理的支持区域
%   residual     : single，Mag_registered - Mag_denoised
%   report       : 每个切片的简要统计
%
% 实现说明
%   1) 每个像素取一个 patch_size x patch_size x 10 的局部块；
%   2) 重排为 10 x Npatch 矩阵；
%   3) 对 10 x 10 协方差矩阵做特征分解；
%   4) 使用 Veraart 等提出的 Marchenko-Pastur 判据估计噪声方差
%      和信号秩；
%   5) 仅将该 patch 的中心像素投影到信号子空间并回写。
%
% 参考
%   Veraart J, et al. Denoising of diffusion MRI using random matrix
%   theory. NeuroImage. 2016;142:394-406.
%   Does MD, et al. Evaluation of PCA image denoising on multi-exponential
%   MRI relaxometry. Magn Reson Med. 2019;81:3503-3514.
%
% 注意
%   本函数面向非商业科研，不用于临床诊断。

    narginchk(1, 2);
    if nargin < 2 || isempty(opts)
        opts = struct();
    end
    opts = apply_default_options(opts);

    input_was_3d = (ndims(Mag_registered) == 3);
    if input_was_3d
        Mag_registered = reshape(Mag_registered, ...
            size(Mag_registered, 1), size(Mag_registered, 2), ...
            size(Mag_registered, 3), 1);
    end

    assert(ndims(Mag_registered) == 4, ...
        '输入必须为 [Nx, Ny, Nweight, Nslice]。');
    assert(isnumeric(Mag_registered) && isreal(Mag_registered), ...
        '当前实现要求实数 magnitude 图像。');

    [nx, ny, n_weight, n_slice] = size(Mag_registered);
    assert(n_weight == 10, ...
        '本项目期望 10 张权重图像，当前第三维为 %d。', n_weight);

    patch_size = opts.patch_size;
    assert(numel(patch_size) == 2 && all(patch_size >= 3), ...
        'patch_size 应为两个不小于 3 的整数。');
    assert(all(mod(patch_size, 2) == 1), ...
        'patch_size 必须为奇数，例如 [5 5]。');

    input_class = class(Mag_registered);
    Mag_single = single(Mag_registered);
    Mag_single(~isfinite(Mag_single)) = 0;

    den_single = zeros(size(Mag_single), 'single');
    sigma_map = zeros(nx, ny, n_slice, 'single');
    rank_map = zeros(nx, ny, n_slice, 'uint8');
    support_mask = false(nx, ny, n_slice);
    report_cell = cell(1, n_slice);

    can_parallel = false;
    if opts.use_parallel
        try
            can_parallel = license('test', 'Distrib_Computing_Toolbox');
        catch
            can_parallel = false;
        end
    end

    if can_parallel
        fprintf('使用 parfor 逐切片处理，共 %d 个切片。\n', n_slice);
        parfor s = 1:n_slice
            [den_s, sigma_s, rank_s, mask_s, report_s] = ...
                denoise_one_slice(Mag_single(:, :, :, s), opts);
            den_single(:, :, :, s) = den_s;
            sigma_map(:, :, s) = sigma_s;
            rank_map(:, :, s) = rank_s;
            support_mask(:, :, s) = mask_s;
            report_cell{s} = report_s;
        end
    else
        fprintf('使用普通 for 循环逐切片处理，共 %d 个切片。\n', n_slice);
        for s = 1:n_slice
            fprintf('  slice %d / %d ...\n', s, n_slice);
            [den_s, sigma_s, rank_s, mask_s, report_s] = ...
                denoise_one_slice(Mag_single(:, :, :, s), opts);
            den_single(:, :, :, s) = den_s;
            sigma_map(:, :, s) = sigma_s;
            rank_map(:, :, s) = rank_s;
            support_mask(:, :, s) = mask_s;
            report_cell{s} = report_s;
        end
    end

    if opts.clip_negative
        den_single = max(den_single, 0);
    end

    residual = Mag_single - den_single;
    Mag_denoised = cast(den_single, input_class);
    report = [report_cell{:}];

    if input_was_3d
        Mag_denoised = Mag_denoised(:, :, :, 1);
        sigma_map = sigma_map(:, :, 1);
        rank_map = rank_map(:, :, 1);
        support_mask = support_mask(:, :, 1);
        residual = residual(:, :, :, 1);
    end
end

function [den, sigma_img, rank_img, mask, report] = denoise_one_slice(data, opts)
% data: [Nx, Ny, Nweight]

    [nx, ny, n_weight] = size(data);
    px = opts.patch_size(1);
    py = opts.patch_size(2);
    rx = floor(px / 2);
    ry = floor(py / 2);

    % 与原 mapping 程序相似的公共支持区域。
    mean_img = mean(data, 3);
    threshold = mean(mean_img(:)) * opts.mask_threshold_fraction;
    mask = isfinite(mean_img) & (mean_img > threshold);

    if opts.min_component_size > 0 && exist('bwareaopen', 'file') == 2
        mask = bwareaopen(mask, opts.min_component_size);
    end
    if opts.fill_holes && exist('imfill', 'file') == 2
        mask = imfill(mask, 'holes');
    end

    den = data;  % 支持区域外保持原始值
    sigma_img = zeros(nx, ny, 'single');
    rank_img = zeros(nx, ny, 'uint8');

    % 对称填充，使边缘像素也能获得完整 patch。
    if exist('padarray', 'file') == 2
        padded = padarray(data, [rx, ry, 0], 'symmetric', 'both');
    else
        error(['需要 Image Processing Toolbox 中的 padarray。', ...
               '你的原流程已使用 bwareaopen，因此通常已安装该工具箱。']);
    end

    center_col = sub2ind([px, py], rx + 1, ry + 1);
    process_idx = find(mask);

    sigma_values = nan(numel(process_idx), 1, 'single');
    rank_values = zeros(numel(process_idx), 1, 'uint8');
    relative_change = zeros(numel(process_idx), 1, 'single');

    for q = 1:numel(process_idx)
        [ix, iy] = ind2sub([nx, ny], process_idx(q));

        % pad 后，原图 (ix,iy) 周围 patch 对应 ix:ix+2rx, iy:iy+2ry。
        patch = padded(ix:ix + 2*rx, iy:iy + 2*ry, :);
        X = reshape(permute(patch, [3, 1, 2]), n_weight, []);

        if any(~isfinite(X(:)))
            continue;
        end

        [x_center_dn, sigma_here, rank_here, ok] = ...
            mppca_center_vector(X, center_col, opts.center_data);

        if ~ok
            continue;
        end

        if opts.clip_negative
            x_center_dn = max(x_center_dn, 0);
        end

        x_center_raw = reshape(data(ix, iy, :), n_weight, 1);
        den(ix, iy, :) = reshape(single(x_center_dn), 1, 1, n_weight);
        sigma_img(ix, iy) = single(sigma_here);
        rank_img(ix, iy) = uint8(rank_here);

        sigma_values(q) = single(sigma_here);
        rank_values(q) = uint8(rank_here);
        relative_change(q) = single(norm(x_center_raw - x_center_dn) / ...
            max(norm(x_center_raw), eps('single')));
    end

    valid_sigma = sigma_values(isfinite(sigma_values) & sigma_values > 0);
    valid_rank = double(rank_values(rank_values > 0));
    valid_change = relative_change(isfinite(relative_change));

    report = struct();
    report.processed_pixels = numel(process_idx);
    report.mask_fraction = nnz(mask) / numel(mask);
    report.threshold = threshold;

    if isempty(valid_sigma)
        report.median_sigma = NaN;
        report.mean_sigma = NaN;
    else
        report.median_sigma = median(double(valid_sigma));
        report.mean_sigma = mean(double(valid_sigma));
    end

    if isempty(valid_rank)
        report.median_rank = NaN;
        report.mean_rank = NaN;
    else
        report.median_rank = median(valid_rank);
        report.mean_rank = mean(valid_rank);
    end

    if isempty(valid_change)
        report.median_relative_change = NaN;
        report.mean_relative_change = NaN;
    else
        report.median_relative_change = median(double(valid_change));
        report.mean_relative_change = mean(double(valid_change));
    end
end

function [x_center_dn, sigma, n_signal, ok] = ...
    mppca_center_vector(X, center_col, center_data)
% 对一个 [M measurements x N spatial samples] patch 做 MP-PCA，
% 仅返回中心空间样本的去噪 M 维向量。
%
% MP 阈值部分遵循 Veraart 等公开实现中的双噪声方差估计判据。

    [M, N] = size(X);
    ok = false;
    sigma = NaN;
    n_signal = 0;
    x_center_dn = X(:, center_col);

    if M < 2 || N < 2 || center_col < 1 || center_col > N
        return;
    end

    Xwork = double(X);

    if center_data
        % 与 Veraart MP.m 的 centering 选项一致：对每个空间样本，
        % 去除其在 measurement 维度上的均值。
        col_mean = mean(Xwork, 1);
        Xwork = Xwork - repmat(col_mean, M, 1);
        center_offset = col_mean(center_col);
        centering_dof = 1;
    else
        center_offset = 0;
        centering_dof = 0;
    end

    R = min(M, N);
    R_eff = R - centering_dof;
    if R_eff < 1
        return;
    end

    % 只需分解 M x M 协方差矩阵，比对每个 patch 直接做完整 SVD 更省时。
    C = (Xwork * Xwork.') / N;
    C = (C + C.') / 2;
    [U, D] = eig(C, 'vector');
    [vals, order] = sort(real(D), 'descend');
    U = real(U(:, order));
    vals = max(vals, 0);

    vals_eff = vals(1:R_eff);

    % 第一种噪声方差估计：候选尾部特征值的均值，并考虑 M>N 时的缩放。
    scaling = ones(R_eff, 1);
    if M > N
        scaling = (M - (0:R_eff-1))' / N;
        scaling(scaling < 1) = 1;
    end

    csum = cumsum(vals_eff(end:-1:1));
    cmean = csum(end:-1:1) ./ (R_eff:-1:1)';
    sigmasq_1 = cmean ./ scaling;

    % 第二种噪声方差估计：观测谱宽与 MP 理论谱宽比较。
    gamma = (M - (0:R_eff-1))' / N;
    range_mp = 4 * sqrt(gamma);
    range_data = vals_eff - vals_eff(end);
    sigmasq_2 = range_data ./ range_mp;

    % 第一个满足条件的位置 t：1..t-1 为信号分量，t..末尾为噪声分量。
    t = find(sigmasq_2 < sigmasq_1, 1, 'first');

    if isempty(t) || ~isfinite(sigmasq_1(min(t, numel(sigmasq_1))))
        return;
    end

    sigma = sqrt(max(sigmasq_1(t), 0));
    n_signal = t - 1;

    x_center = Xwork(:, center_col);
    if n_signal > 0
        U_signal = U(:, 1:n_signal);
        x_center_dn = U_signal * (U_signal.' * x_center);
    else
        x_center_dn = zeros(M, 1);
    end

    x_center_dn = x_center_dn + center_offset;
    ok = all(isfinite(x_center_dn)) && isfinite(sigma);
end

function opts = apply_default_options(opts)
    defaults = struct();
    defaults.patch_size = [5, 5];
    defaults.center_data = false;
    defaults.clip_negative = true;
    defaults.use_parallel = true;
    defaults.mask_threshold_fraction = 1/5;
    defaults.min_component_size = 20;
    defaults.fill_holes = false;
    defaults.save_residual = true;
    defaults.overwrite = false;

    names = fieldnames(defaults);
    for k = 1:numel(names)
        name = names{k};
        if ~isfield(opts, name) || isempty(opts.(name))
            opts.(name) = defaults.(name);
        end
    end
end
