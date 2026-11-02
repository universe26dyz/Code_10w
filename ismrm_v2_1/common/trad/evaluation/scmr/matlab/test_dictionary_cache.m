function test_dictionary_cache()
% Synthetic exact-equivalence: uncached legacy matcher, cache miss, cache hit.
    here=fileparts(mfilename('fullpath')); utility=fullfile(fileparts(fileparts(fileparts(here))),'modules','module_01_preprocess_matlab','hhz_original','Utility'); addpath(here); addpath(utility);
    info=struct('RepetitionTime',2.61,'EchoTrainLength',87); acq=(0:9)'*900; FA=[45 45 45]; TI=[50 150]; T2=[35 45 55]; Tlist=[1000 50 1];
    dictionary=build_multimap_dictionary(info,acq,FA,Tlist,T2,TI); mag=repmat(reshape(dictionary.dictionary(1,:),1,1,10),24,24,1);
    [u1,u2,u3]=function_T1T2_10HB_bssfp(mag,info,acq,FA,Tlist,T2,TI);
    root=tempname; mkdir(root); sim=fullfile(utility,'sim_T1T2_10HB_bssfp.m');
    [first,p1]=get_or_build_multimap_dictionary(info,acq,FA,Tlist,T2,TI,root,sim); [c1,c2,c3,v1]=match_multimap_dictionary(mag,first);
    [second,p2]=get_or_build_multimap_dictionary(info,acq,FA,Tlist,T2,TI,root,sim); [d1,d2,d3,v2]=match_multimap_dictionary(mag,second);
    assert(strcmp(p1.cache_status,'miss') && strcmp(p2.cache_status,'hit')); assert(isequal(Tlist,first.Tlist) && isequal(first.dict_norm,second.dict_norm));
    assert(isequal(u1,c1) && isequal(u2,c2) && isequal(u3,c3)); assert(isequal(c1,d1) && isequal(c2,d2) && isequal(c3,d3) && isequal(v1,v2)); rmdir(root,'s');
end
