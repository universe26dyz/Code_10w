function [T1map, T2map, B1map, qc] = function_T1T2_10HB_bssfp_qc(mag, info, Acq_time, FA, Tlist, T2_prep, TI)
%FUNCTION_T1T2_10HB_BSSFP_QC SCMR-v5 10-heartbeat dictionary matching.
% This is the exact matcher used to reproduce the retained HHZ SCMR-v5
% mapping contract.  Keep this copy beside the Code_10w wrapper so the
% server run cannot accidentally resolve an unrelated legacy implementation.

n_ex = info.EchoTrainLength;
TR = info.RepetitionTime;
n_images = size(mag, 3);
if n_images ~= 10
    error('function_T1T2_10HB_bssfp_qc expects 10 images, but got %d.', n_images);
end

img_mean = mean(mag, 3);
mask = img_mean;
mask(mask < mean(mask(:))/5) = 0;
mask(mask > 0) = 1;
mask = bwareaopen(mask, 20);
sig = shiftdim(mag, 2);
sig = sig(:, mask > 0);

PP_delay = zeros(n_images, 3);
PP_delay(1,1) = TI(1); PP_delay(5,1) = TI(2);
PP_delay(8,2) = T2_prep(1); PP_delay(9,2) = T2_prep(2); PP_delay(10,2) = T2_prep(3);
Duration_befor_Acq = zeros(n_images,1);
for i = 2:n_images
    Duration_befor_Acq(i) = Acq_time(i) - Acq_time(i-1) - TR*(n_ex + 10);
end
Duration_befor_Acq(5) = Duration_befor_Acq(5) - PP_delay(5,1);
Duration_befor_Acq(8) = Duration_befor_Acq(8) - PP_delay(8,2);
Duration_befor_Acq(9) = Duration_befor_Acq(9) - PP_delay(9,2);
Duration_befor_Acq(10) = Duration_befor_Acq(10) - PP_delay(10,2);

dictionary = zeros(size(Tlist,1), n_images);
parfor ind = 1:size(Tlist,1)
    pars = reshape(Tlist(ind,:), 1, []);
    dictionary(ind,:) = abs(sim_T1T2_10HB_bssfp(n_ex, TR, pars, n_images, ...
        PP_delay, Duration_befor_Acq, FA));
end
dictionary = abs(dictionary);
dict_norm = zeros(size(dictionary), 'single');
for i = 1:size(dictionary,1)
    nrm = norm(dictionary(i,:));
    if nrm ~= 0
        dict_norm(i,:) = single(dictionary(i,:) ./ nrm);
    end
end

sig_norm = zeros(size(sig), 'single');
valid_sig = false(1, size(sig,2));
for i = 1:size(sig,2)
    nrm = norm(sig(:,i));
    if nrm > 0
        sig_norm(:,i) = single(sig(:,i) ./ nrm);
        valid_sig(i) = true;
    end
end

Npix = size(sig_norm, 2);
match_score = zeros(1, Npix, 'single');
second_score = zeros(1, Npix, 'single');
matchout = zeros(1, Npix, 'uint32');
fit_nrmse = zeros(1, Npix, 'single');
block_size = 200;
for start_idx = 1:block_size:Npix
    end_idx = min(start_idx + block_size - 1, Npix);
    scores = abs(dict_norm * sig_norm(:, start_idx:end_idx));
    [score_here, idx_here] = max(scores, [], 1);
    linear_best = sub2ind(size(scores), double(idx_here), 1:size(scores,2));
    scores(linear_best) = -inf;
    second_here = max(scores, [], 1);
    match_score(start_idx:end_idx) = score_here;
    second_score(start_idx:end_idx) = second_here;
    matchout(start_idx:end_idx) = uint32(idx_here);
    best_dict = dict_norm(double(idx_here), :);
    residual = sig_norm(:, start_idx:end_idx)' - best_dict;
    fit_nrmse(start_idx:end_idx) = single(sqrt(sum(residual.^2, 2))' / sqrt(n_images));
end

T1list = Tlist(double(matchout(:)),1);
T2list = Tlist(double(matchout(:)),2);
B1list = Tlist(double(matchout(:)),3);
score_th = 0.90;
sig_amp = max(sig, [], 1);
amp_th = 0.05 * max(sig_amp);
valid_match = (match_score > score_th) & (sig_amp > amp_th) & valid_sig;
T1list(~valid_match(:)) = 0; T2list(~valid_match(:)) = 0; B1list(~valid_match(:)) = 0;

T1map = zeros(size(mask)); T2map = zeros(size(mask)); B1map = zeros(size(mask));
T1map(mask > 0) = T1list; T2map(mask > 0) = T2list; B1map(mask > 0) = B1list;
qc.match_score_map = zeros(size(mask), 'single');
qc.second_best_score_map = zeros(size(mask), 'single');
qc.match_margin_map = zeros(size(mask), 'single');
qc.valid_match_map = false(size(mask));
qc.signal_amplitude_map = zeros(size(mask), 'single');
qc.fitting_nrmse_map = zeros(size(mask), 'single');
qc.best_index_map = zeros(size(mask), 'uint32');
qc.match_score_map(mask > 0) = match_score;
qc.second_best_score_map(mask > 0) = second_score;
qc.match_margin_map(mask > 0) = match_score - second_score;
qc.valid_match_map(mask > 0) = valid_match;
qc.signal_amplitude_map(mask > 0) = single(sig_amp);
qc.fitting_nrmse_map(mask > 0) = fit_nrmse;
qc.best_index_map(mask > 0) = matchout;
qc.score_threshold = score_th;
qc.amplitude_threshold = amp_th;
qc.nrmse_definition = 'norm(s_norm-d_best_norm,2)/sqrt(10)';
qc.match_margin_definition = 'best_score-second_best_score';
qc.method_version = 'v5';
qc.nr_startups = 10;
qc.duration_before_acq_ms = Duration_befor_Acq;
qc.timing_formula = 'delta_AcquisitionTime-TR*(EchoTrainLength+10)-preparation_delay';
end
