function [T1map, T2map, B1map] = function_T1T2_10HB_bssfp(mag, info, Acq_time, FA, Tlist, T2_prep, TI)

% function_T1T2_10HB_bssfp 基于10幅bSSFP图像和字典匹配，定量计算T1、T2和B1图
% 输入：
%   mag      - [Nx, Ny, 10] 复数或实数图像，每幅对应不同的准备脉冲状态
%   info     - 结构体，包含 EchoTrainLength (每TR内的回波数) 和 RepetitionTime (TR)
%   Acq_time - 长度为10的向量，每幅图像的实际采集时刻（相对于序列起点）
%   FA       - 翻转角（标量或向量，本例中可能用于模拟）
%   Tlist    - [M, 3] 字典参数矩阵，每行 [T1, T2, B1] 候选值
%   T2_prep  - 长度为3的向量，T2准备脉冲的等效TE时间（用于第8、9、10幅图像）
%   TI       - 长度为2的向量，反转恢复延迟（用于第1和第5幅图像）
% 输出：
%   T1map, T2map, B1map - [Nx, Ny] 浮点矩阵，定量参数图


% ----- 1. 基本信息提取与校验 -----
n_ex = info.EchoTrainLength;           % 每个TR内的回波链长度
TR = info.RepetitionTime;              % 重复时间 (ms)
n_images = size(mag, 3);               % 图像数量应为10

if n_images ~= 10
    error('sim_T1T2_10HB expects 10 images, but got %d.', n_images);
end

% ----- 2. 生成有效像素掩膜（去除背景） -----
img_mean = mean(mag, 3);               % 对所有图像求平均
mask = img_mean;
mask(mask < mean(mask(:))/5) = 0;      % 阈值：低于整体均值1/5的设为0
mask(mask > 0) = 1;                    % 二值化
mask = bwareaopen(mask, 20);           % 去除小于20个像素的小连通区域（去噪）

% ----- 3. 信号重排：将时间维移至第一维，并提取有效像素 -----
sig = shiftdim(mag, 2);                % 尺寸变为 [10, Nx, Ny]
sig = sig(:, mask > 0);                % 只保留掩膜内的像素，尺寸 [10, Npix]

% timing model
% ----- 4. 时序模型：准备脉冲（反转、T2准备）的延迟修正 -----
PP_delay = zeros(10, 2);               % 10幅图像，每行 [反演延迟, T2准备TE]

% column 1: inversion delay
% 第1幅图像有反演延迟 TI(1)（在信号模拟中起效）
PP_delay(1,1) = TI(1);
% 第5幅图像有反演延迟 TI(2)
PP_delay(5,1) = TI(2);

% column 2: T2prep TE
% 第8、9、10幅图像有T2准备脉冲，分别对应T2_prep的三个值
PP_delay(8,2) = T2_prep(1);
PP_delay(9,2) = T2_prep(2);
PP_delay(10,2) = T2_prep(3);

% 计算每幅图像之间的采集时间间隔（不考虑回波链内的TR累积）
Duration_befor_Acq = zeros(n_images,1);
for i = 2:n_images
    Duration_befor_Acq(i) = Acq_time(i) - Acq_time(i-1) - TR*(n_ex);
end
% 注意：Duration_befor_Acq(1)始终为0，因为它表示第一幅图之前的持续时间

% 剔除准备脉冲本身占用的时间，这些时间在模拟中被单独处理
Duration_befor_Acq(5)  = Duration_befor_Acq(5)  - PP_delay(5,1);
Duration_befor_Acq(8)  = Duration_befor_Acq(8)  - PP_delay(8,2);
Duration_befor_Acq(9)  = Duration_befor_Acq(9)  - PP_delay(9,2);
Duration_befor_Acq(10) = Duration_befor_Acq(10) - PP_delay(10,2);
% 修正后，Duration_befor_Acq(i)表示第i-1幅到第i幅之间的纯弛豫等待时间

% ----- 5. 构建字典（信号模拟） -----
disp('begin build dict')
tic

dictionary = zeros(size(Tlist,1), n_images);   % [字典条目数, 10]

% 并行遍历每个参数组合，模拟对应的10个信号值
parfor ind = 1:size(Tlist,1)
    pars = reshape(Tlist(ind,:), 1, []);       % 当前 [T1, T2, B1]
    % 调用仿真函数（该子函数需要实现bSSFP序列的信号演化）
    sim_im_val = sim_T1T2_10HB_bssfp(n_ex, TR, pars, n_images, PP_delay, Duration_befor_Acq, FA);
    dictionary(ind,:) = abs(sim_im_val);       % 取模（信号幅度）
end

% ----- 6. 字典匹配（基于归一化内积） -----
disp('begin matching')
tic

dictionary = abs(dictionary);                  % 确保幅值

% 6.1 归一化字典：将每行（每个参数组合）归一化为单位向量
dict_norm = zeros(size(dictionary,1), n_images, 'single');
for i = 1:size(dictionary,1)
    nrm = norm(dictionary(i,:));
    if nrm ~= 0
        dict_norm(i,:) = single(dictionary(i,:) ./ nrm);
    end
end

% 6.2 归一化信号：每个像素的10个时间点信号归一化
sig_norm = zeros(size(sig,1), size(sig,2), 'single');
valid_sig = false(1, size(sig,2));
for i = 1:size(sig,2)
    nrm = norm(sig(:,i));
    if nrm > 0
        sig_norm(:,i) = single(sig(:,i) ./ nrm);
        valid_sig(i) = true;
    end
end

% 6.3 分块匹配，避免内存溢出
Npix = size(sig_norm, 2);
match_score = zeros(1, Npix, 'single');        % 存储最佳匹配分数
matchout = zeros(1, Npix, 'uint32');           % 存储最佳匹配索引

block_size = 200;   % 可调节，取决于可用内存

for start_idx = 1:block_size:Npix
    end_idx = min(start_idx + block_size - 1, Npix);

    % 计算当前块所有像素与所有字典条目的内积（相似度）
    % innerproduct 尺寸为 [M, block_size]
    innerproduct = dict_norm * sig_norm(:, start_idx:end_idx);
    % 取绝对值最大者作为匹配得分和索引
    [score_here, idx_here] = max(abs(innerproduct), [], 1);

    match_score(start_idx:end_idx) = score_here;
    matchout(start_idx:end_idx) = uint32(idx_here);

    disp(['matching pixels ' num2str(start_idx) ' to ' num2str(end_idx) ' / ' num2str(Npix)])
end

% ----- 7. 根据匹配索引提取T1、T2、B1值 -----
T1list = Tlist(double(matchout(:)),1);
T2list = Tlist(double(matchout(:)),2);
B1list = Tlist(double(matchout(:)),3);

% ----- 8. 质量过滤 -----
score_th = 0.90;                         % 匹配分数阈值
sig_amp = max(sig, [], 1);               % 每个像素的信号最大幅度
amp_th = 0.05 * max(sig_amp);            % 幅度阈值（相对于全局最大幅度的5%）

% 有效像素需同时满足：匹配分数高、信号幅度足够、且归一化成功
valid_match = (match_score > score_th) & (sig_amp > amp_th) & valid_sig;

% 将无效像素的参数置为0（通常表示背景或不可靠区域）
T1list(~valid_match(:)) = 0;
T2list(~valid_match(:)) = 0;
B1list(~valid_match(:)) = 0;

% ----- 9. 将一维结果映射回二维图像 -----
T1map = zeros(size(mask));
T2map = zeros(size(mask));
B1map = zeros(size(mask));

T1map(mask > 0) = T1list;
T2map(mask > 0) = T2list;
B1map(mask > 0) = B1list;

toc

end
