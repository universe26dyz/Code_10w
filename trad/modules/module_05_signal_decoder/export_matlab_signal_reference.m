function export_matlab_signal_reference(output_mat)
% Deterministic raw-HHZ reference only for the Phase 3 parity check.
if nargin ~= 1
    error('Exactly one output MAT path is required.');
end
this_dir = fileparts(mfilename('fullpath'));
utility_dir = fullfile(this_dir, '..', 'module_01_preprocess_matlab', 'hhz_original', 'Utility');
addpath(utility_dir);

n_cases = 20;
n_ex = 32;
TR = 3.2;
FA = [45, 45, 45];
T1 = linspace(550, 1850, n_cases)';
T2 = linspace(20, 130, n_cases)';
B1 = linspace(0.25, 1.15, n_cases)';
timing9_ms = repmat([75, 82, 91, 68, 84, 77, 65, 59, 53], n_cases, 1) ...
    + (0:n_cases-1)' * [0.5, 0.25, 0.75, 0.5, 0.25, 0.75, 0.5, 0.25, 0.5];
PP_delay = zeros(10, 2);
PP_delay(1, 1) = 50;
PP_delay(5, 1) = 150;
PP_delay(8, 2) = 35;
PP_delay(9, 2) = 45;
PP_delay(10, 2) = 55;
signal_raw = zeros(n_cases, 10);
for k = 1:n_cases
    duration = [0; timing9_ms(k, :)'];
    signal_raw(k, :) = sim_T1T2_10HB_bssfp(n_ex, TR, [T1(k), T2(k), B1(k)], 10, PP_delay, duration, FA);
end
save(output_mat, 'T1', 'T2', 'B1', 'timing9_ms', 'n_ex', 'TR', 'FA', 'signal_raw');
end
