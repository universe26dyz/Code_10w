% Check_MPPCA_Result.m
% 快速检查 raw / MP-PCA / residual，避免去噪残差中出现明显解剖结构。

close all; clc; clear;

%% 用户配置
file_name = '2ch_data_all_mppca_7c.mat';
slice_index = 5;     % 空间切片编号
weight_index = 9;    % 1~10

S = load(file_name);
assert(isfield(S, 'Mag_crop_raw') && isfield(S, 'Mag_crop'), ...
    '文件中需要同时包含 Mag_crop_raw 与 Mag_crop。');

raw = double(S.Mag_crop_raw(:, :, weight_index, slice_index));
den = double(S.Mag_crop(:, :, weight_index, slice_index));
res = raw - den;

figure('Name', 'MP-PCA quality check', 'Color', 'w');
subplot(1, 3, 1);
imagesc(raw); axis image off; colorbar;
title(sprintf('Raw | slice %d | weight %d', slice_index, weight_index));

subplot(1, 3, 2);
imagesc(den); axis image off; colorbar;
title('MP-PCA');

subplot(1, 3, 3);
lim = max(abs(res(:)));
if lim == 0
    lim = 1;
end
imagesc(res, [-lim, lim]); axis image off; colorbar;
title('Residual = Raw - MP-PCA');
colormap gray;

fprintf('Raw mean/std: %.6g / %.6g\n', mean(raw(:)), std(raw(:)));
fprintf('Denoised mean/std: %.6g / %.6g\n', mean(den(:)), std(den(:)));
fprintf('Residual RMS: %.6g\n', sqrt(mean(res(:).^2)));
