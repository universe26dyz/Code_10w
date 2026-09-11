function [sim_im_val] = sim_T1T2_10HB_bssfp(n_ex, TR, Tlist, n_img, PP_delay, Duration_befor_Acq, FA)
% ind  Rwave
%  1   |...Inv....ACQ...
%  2   |..........ACQ...
%  3   |..........ACQ...
%  4   |..........ACQ...

%  5   |...Inv....ACQ...
%  6   |..........ACQ...
%  7   |..........ACQ...

%  8   |......T2p1ACQ...
%  9   |......T2p2ACQ...
%  10  |......T2p3ACQ...

% 函数功能：模拟10幅平衡稳态自由进动(bSSFP)图像在给定组织参数下的信号强度
% 应用场景：心脏磁共振多对比度bSSFP序列仿真，支持反转准备、T2准备等多种准备模块
%
% 输入参数：
%   n_ex              - 每幅图像采集块内的回波链长度(ETL)，即采集阶段重复TR的次数
%   TR                - 射频重复时间(单位：ms)
%   Tlist             - 组织参数向量 [T1, T2, B1]
%                       T1: 纵向弛豫时间(ms)；T2: 横向弛豫时间(ms)；B1: B1场校正系数(射频场不均匀性)
%   n_img             - 待模拟的图像总数，本函数固定为10
%   PP_delay          - 10×2 准备脉冲参数矩阵，每行对应一幅图像
%                       第1列：反转脉冲后的延迟时间(ms)；第2列：T2准备的回波时间TE(ms)
%   Duration_befor_Acq- 长度为10的向量，每幅图像采集前的纯纵向弛豫等待时间
%                       已扣除准备脉冲时长和采集块时长，仅包含自由弛豫阶段
%   FA                - 三元素翻转角向量(单位：度)，分三段对应不同图像的标称翻转角
%                       FA(1): 第1~4幅图像；FA(2): 第5~7幅图像；FA(3): 第8~10幅图像
%
% 输出参数：
%   sim_im_val        - 1×10 向量，模拟得到的10幅图像的信号幅度
%                       取每幅图像采集块中间区域回波的平均信号，对应K空间中心的对比度
%
% 说明：本函数按序列时间轴依次处理10幅图像，采用简化二分量EPG模型模拟bSSFP稳态信号
%       每幅图像依次经过：前置弛豫 → 准备脉冲(反转/T2准备) → bSSFP采集段 → 输出信号

%% -------------------------- 1. 参数提取与初始化 --------------------------
T1 = Tlist(1);   % 提取组织纵向弛豫时间T1
T2 = Tlist(2);   % 提取组织横向弛豫时间T2
B1 = Tlist(3);   % 提取B1场校正系数，用于修正实际翻转角

nRampUp = 10;    % bSSFP采集前的翻转角爬升脉冲数，用于快速建立稳态
npulse = n_ex;   % 采集块内的总射频脉冲数，等于回波链长度

Mz = 1;          % 初始纵向磁化矢量，归一化平衡磁化M0=1
M0 = 1;          % 平衡态纵向磁化强度，归一化值为1
sim_im_val = zeros(1, n_img);  % 初始化输出信号数组

%% -------------------------- 2. 逐幅图像循环模拟 --------------------------
for ii = 1 : n_img
    
    % --- 2.1 根据图像序号分配翻转角，并做B1场校正 ---
    if ii <= 4
        flip_angle = FA(1) * B1;   % 第1-4幅：使用第一组翻转角，B1校正实际角度
    elseif ii >=5 && ii <=7
        flip_angle = FA(2) * B1;   % 第5-7幅：使用第二组翻转角
    elseif ii >=8 && ii <=10
        flip_angle = FA(3) * B1;   % 第8-10幅：使用第三组翻转角
    end
    alpha = d2r(flip_angle);       % 翻转角从角度转换为弧度
%     ss = flip_angle * ones([1 npulse]);
% 
%     alpha=d2r(ss);

    % --- 2.2 采集前纯纵向弛豫阶段 ---
    %%% T1 recovery
    if (Duration_befor_Acq(ii) ~= 0)
        TI_here=Duration_befor_Acq(ii);   % Spoiler brefor readout with duration 6mS
        % 纵向弛豫恢复公式：Mz(t) = M0*(1-exp(-t/T1)) + Mz0*exp(-t/T1)
        Mz=M0*(1-exp(-TI_here/T1))+Mz*exp(-TI_here/T1);
    end

    % --- 2.3 反转脉冲 + 反转后T1恢复阶段 ---
    %%% Inv & T1 recovery
    if(PP_delay(ii,1) ~= 0)
        Mz=-Mz;%*exp(-4/T2); % the excess T2 decay during Inversion pulse%MQF0.85% 理想180°反转脉冲：纵向磁化矢量直接反号
        TI_here=PP_delay(ii,1);% 反转后的延迟时间(TI)
        Mz=M0*(1-exp(-TI_here/T1))+Mz*exp(-TI_here/T1);% 反转后纵向弛豫恢复
    end

    % --- 2.4 T2准备脉冲阶段：横向弛豫衰减 ---
    %%% T2 decay
    if(PP_delay(ii,2) ~= 0)
        TE_here=PP_delay(ii,2);% T2准备的回波时间TE
        % T2弛豫衰减：横向磁化按exp(-TE/T2)衰减
        Mz=Mz*exp(-TE_here/T2);
%         Mz=T2prep_Simu(T1,T2,TE_here,B1,Mz);
    end

% --- 以下为注释掉的可选模块：T1rho准备、采集前扰相 ---
    % % T1rho弛豫衰减模块
    % if(PP_delay(ii,3) ~= 0)
    %     TSL_here = PP_delay(ii,3);
    %     Mz = Mz*exp(-TSL_here/T1rho)*exp(-3.6/T2);
    % end
    % % 采集前扰相梯度恢复模块
    % TI_here=6;
    % Mz=M0*(1-exp(-TI_here/T1))+Mz*exp(-TI_here/T1);

%% -------------------------- 3. bSSFP采集段信号模拟 --------------------------
    sig = [0; Mz];   % 二分量磁化矢量：[横向磁化Mx; 纵向磁化Mz]，初始横向磁化为0
    E1 = exp(-TR/T1);% 单个TR内的纵向弛豫因子
    E2 = exp(-TR/T2);% 单个TR内的横向弛豫因子
    Dirc = 1;        % bSSFP翻转角方向标记：交替正负，初始为正方向

    % --- 3.1 翻转角爬升段(Ramp-Up)：逐步增加翻转角，快速建立稳态 ---
    for n = 1 : nRampUp
        % 第n个爬升脉冲的实际翻转角：线性递增，交替正负方向
        AlphaTmp = d2r(flip_angle / nRampUp * n) * Dirc;
        
        % 射频脉冲作用：绕x轴的旋转矩阵，更新横向+纵向磁化
        sig = [cos(AlphaTmp), -sin(AlphaTmp); 
               sin(AlphaTmp),  cos(AlphaTmp)] * sig;
        
        % TR间期的弛豫过程：横向T2衰减，纵向T1恢复
        sig(1) = sig(1) * E2;                  % 横向磁化T2衰减
        sig(2) = 1 + (sig(2) - 1) * E1;        % 纵向磁化T1恢复(向平衡态M0=1趋近)
        
        Dirc = -1 * Dirc;   % 翻转角方向取反，bSSFP交替正负脉冲特性
    end

    % --- 3.2 稳态采集段：完整翻转角的bSSFP回波链 ---
    sig_xy = [];   % 存储每个回波的横向信号幅度
    for n = 1 : npulse
        AlphaTmp = alpha * Dirc;   % 稳态翻转角，交替正负方向
        
        % 射频脉冲作用：旋转矩阵更新磁化矢量
        sig = [cos(AlphaTmp), -sin(AlphaTmp); 
               sin(AlphaTmp),  cos(AlphaTmp)] * sig;
        
        % 记录当前回波的横向磁化幅度（取绝对值为信号强度）
        sig_xy = [sig_xy, abs(sig(1))];
        
        % TR间期弛豫
        sig(1) = sig(1) * E2;                  % 横向T2衰减
        sig(2) = 1 + (sig(2) - 1) * E1;        % 纵向T1恢复
        
        Dirc = -1 * Dirc;   % 翻转角方向交替
    end

%% -------------------------- 4. 计算当前图像信号与状态更新 --------------------------
    % 取回波链中间16个点的平均值作为图像信号（对应K空间中心，代表图像对比度）
    sim_im_val(ii) = mean(sig_xy( round(npulse/2 - 15/2) : round(npulse/2 - 15/2) + 15 ));
    
    % 更新当前采集结束后的纵向磁化，作为下一幅图像的初始状态
    Mz = sig(2);
end

end
