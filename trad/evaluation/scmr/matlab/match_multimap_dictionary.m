function [T1map,T2map,B1map,valid_match] = match_multimap_dictionary(mag, dictionary)
% Exact legacy mask, normalization, inner-product score and thresholds.
    img_mean=mean(mag,3); mask=img_mean; mask(mask<mean(mask(:))/5)=0; mask(mask>0)=1; mask=bwareaopen(mask,20);
    sig=shiftdim(mag,2); sig=sig(:,mask>0); sig_norm=zeros(size(sig,1),size(sig,2),'single'); valid_sig=false(1,size(sig,2));
    for i=1:size(sig,2), nrm=norm(sig(:,i)); if nrm>0, sig_norm(:,i)=single(sig(:,i)./nrm); valid_sig(i)=true; end, end
    Npix=size(sig_norm,2); score=zeros(1,Npix,'single'); out=zeros(1,Npix,'uint32');
    for start=1:200:Npix
        stop=min(start+199,Npix); [s,idx]=max(abs(dictionary.dict_norm*sig_norm(:,start:stop)),[],1); score(start:stop)=s; out(start:stop)=uint32(idx);
    end
    T1list=dictionary.Tlist(double(out(:)),1); T2list=dictionary.Tlist(double(out(:)),2); B1list=dictionary.Tlist(double(out(:)),3);
    amp=max(sig,[],1); valid_match=(score>0.90)&(amp>0.05*max(amp))&valid_sig;
    T1list(~valid_match(:))=0; T2list(~valid_match(:))=0; B1list(~valid_match(:))=0;
    T1map=zeros(size(mask)); T2map=zeros(size(mask)); B1map=zeros(size(mask)); T1map(mask>0)=T1list; T2map(mask>0)=T2list; B1map(mask>0)=B1list;
end
