function dictionary = build_multimap_dictionary(info, Acq_time, FA, Tlist, T2_prep, TI)
% Exact legacy dictionary construction: same timing, simulator, abs and row normalization.
    n_ex = info.EchoTrainLength; TR = info.RepetitionTime; n_images = 10;
    PP_delay = zeros(10,2); PP_delay(1,1)=TI(1); PP_delay(5,1)=TI(2); PP_delay(8:10,2)=T2_prep(:);
    duration = zeros(n_images,1);
    for i=2:n_images, duration(i)=Acq_time(i)-Acq_time(i-1)-TR*n_ex; end
    duration(5)=duration(5)-PP_delay(5,1); duration(8:10)=duration(8:10)-PP_delay(8:10,2);
    values = zeros(size(Tlist,1), n_images);
    parfor ind=1:size(Tlist,1)
        values(ind,:) = abs(sim_T1T2_10HB_bssfp(n_ex, TR, reshape(Tlist(ind,:),1,[]), n_images, PP_delay, duration, FA));
    end
    dict_norm = zeros(size(values,1), n_images, 'single');
    for i=1:size(values,1)
        nrm=norm(values(i,:)); if nrm~=0, dict_norm(i,:)=single(values(i,:)./nrm); end
    end
    dictionary = struct('Tlist',Tlist,'dictionary',values,'dict_norm',dict_norm,'TR',TR,'VPS',n_ex,'Acq_time',Acq_time(:),'FA',FA(:),'TI',TI(:),'T2_prep',T2_prep(:));
end
