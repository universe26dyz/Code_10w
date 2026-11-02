function [sim_im_val] = sim_T1T2_10HB_bssfp(n_ex, TR, Tlist, n_img, PP_delay, Duration_befor_Acq, FA)
T1 = Tlist(1);
T2 = Tlist(2);
B1 = Tlist(3);
nRampUp = 10;
npulse = n_ex;
Mz = 1;
M0 = 1;
sim_im_val = zeros(1, n_img);
for ii = 1:n_img
    if ii <= 4
        flip_angle = FA(1) * B1;
    elseif ii >= 5 && ii <= 7
        flip_angle = FA(2) * B1;
    else
        flip_angle = FA(3) * B1;
    end
    alpha = d2r(flip_angle);
    if Duration_befor_Acq(ii) ~= 0
        TI_here = Duration_befor_Acq(ii);
        Mz = M0*(1-exp(-TI_here/T1)) + Mz*exp(-TI_here/T1);
    end
    if PP_delay(ii,1) ~= 0
        Mz = -Mz;
        TI_here = PP_delay(ii,1);
        Mz = M0*(1-exp(-TI_here/T1)) + Mz*exp(-TI_here/T1);
    end
    if PP_delay(ii,2) ~= 0
        TE_here = PP_delay(ii,2);
        Mz = Mz*exp(-TE_here/T2);
    end
    sig = [0; Mz];
    E1 = exp(-TR/T1);
    E2 = exp(-TR/T2);
    Dirc = 1;
    for n = 1:nRampUp
        AlphaTmp = d2r(flip_angle/nRampUp*n) * Dirc;
        sig = [cos(AlphaTmp), -sin(AlphaTmp); sin(AlphaTmp), cos(AlphaTmp)] * sig;
        sig(1) = sig(1) * E2;
        sig(2) = 1 + (sig(2)-1) * E1;
        Dirc = -1 * Dirc;
    end
    sig_xy = [];
    for n = 1:npulse
        AlphaTmp = alpha * Dirc;
        sig = [cos(AlphaTmp), -sin(AlphaTmp); sin(AlphaTmp), cos(AlphaTmp)] * sig;
        sig_xy = [sig_xy, abs(sig(1))]; %#ok<AGROW>
        sig(1) = sig(1) * E2;
        sig(2) = 1 + (sig(2)-1) * E1;
        Dirc = -1 * Dirc;
    end
    sim_im_val(ii) = mean(sig_xy(round(npulse/2-15/2):round(npulse/2-15/2)+15));
    Mz = sig(2);
end
end
