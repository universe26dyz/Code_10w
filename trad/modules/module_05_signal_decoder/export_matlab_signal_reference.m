function export_matlab_signal_reference(output_mat, config_json)
% Deterministic raw-HHZ reference. config_json is optional for a frozen
% protocol-specific parity gate; one-argument Phase 3 behavior is unchanged.
if nargin < 1 || nargin > 2
    error('One output MAT path and optional JSON config path are required.');
end
this_dir = fileparts(mfilename('fullpath'));
utility_dir = fullfile(this_dir, '..', 'module_01_preprocess_matlab', 'hhz_original', 'Utility');
addpath(utility_dir);

if nargin == 2
    cfg = jsondecode(fileread(config_json));
    T1 = cfg.T1(:); T2 = cfg.T2(:); B1 = cfg.B1(:);
    timing9_ms = cfg.timing9_ms;
    n_ex = cfg.n_ex; TR = cfg.TR; FA = cfg.FA;
else
    n_cases = 20;
    n_ex = 32;
    TR = 3.2;
    FA = [45, 45, 45];
    T1 = linspace(550, 1850, n_cases)';
    T2 = linspace(20, 130, n_cases)';
    B1 = linspace(0.25, 1.15, n_cases)';
    timing9_ms = repmat([75, 82, 91, 68, 84, 77, 65, 59, 53], n_cases, 1) ...
        + (0:n_cases-1)' * [0.5, 0.25, 0.75, 0.5, 0.25, 0.75, 0.5, 0.25, 0.5];
end
if size(timing9_ms, 1) == 1
    timing9_ms = repmat(timing9_ms, numel(T1), 1);
end
if size(timing9_ms, 1) ~= numel(T1) || size(timing9_ms, 2) ~= 9 || numel(T2) ~= numel(T1) || numel(B1) ~= numel(T1)
    error('Reference config must contain matching T1/T2/B1 vectors and [N,9] timing9_ms.');
end
PP_delay = zeros(10, 2);
PP_delay(1, 1) = 50;
PP_delay(5, 1) = 150;
PP_delay(8, 2) = 35;
PP_delay(9, 2) = 45;
PP_delay(10, 2) = 55;
signal_raw = zeros(numel(T1), 10);
for k = 1:numel(T1)
    duration = [0; timing9_ms(k, :)'];
    signal_raw(k, :) = sim_T1T2_10HB_bssfp(n_ex, TR, [T1(k), T2(k), B1(k)], 10, PP_delay, duration, FA);
end
save(output_mat, 'T1', 'T2', 'B1', 'timing9_ms', 'n_ex', 'TR', 'FA', 'signal_raw');
end
