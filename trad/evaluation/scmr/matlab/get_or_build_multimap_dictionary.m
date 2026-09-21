function [dictionary, provenance] = get_or_build_multimap_dictionary(info, Acq_time, FA, Tlist, T2_prep, TI, cache_root, simulator_path)
% Cache is valid only when canonical scientific inputs and simulator hash match.
    schema_version='multimap_dictionary_cache/v1'; sim_hash=multimap_sha256(fileread(simulator_path));
    payload=struct('schema_version',schema_version,'TR',double(info.RepetitionTime),'VPS',double(info.EchoTrainLength),'Acq_time',double(Acq_time(:))','FA',double(FA(:))','TI',double(TI(:))','T2_prep',double(T2_prep(:))','Tlist',double(Tlist),'simulator_sha256',sim_hash,'nRampUp',10);
    signature=multimap_sha256(jsonencode(payload)); provenance=struct('cache_enabled',~isempty(cache_root),'signature',signature,'cache_status','disabled','cache_file','','schema_version',schema_version,'simulator_sha256',sim_hash);
    if isempty(cache_root), dictionary=build_multimap_dictionary(info,Acq_time,FA,Tlist,T2_prep,TI); return; end
    if ~isfolder(cache_root), mkdir(cache_root); end
    cache_file=fullfile(cache_root,[signature '.mat']); provenance.cache_file=cache_file;
    if isfile(cache_file)
        C=load(cache_file,'dictionary','provenance_saved');
        if ~isfield(C,'provenance_saved') || ~strcmp(C.provenance_saved.signature,signature) || ~strcmp(C.provenance_saved.simulator_sha256,sim_hash), error('MultiMap cache provenance mismatch: %s',cache_file); end
        dictionary=C.dictionary; provenance.cache_status='hit'; return;
    end
    dictionary=build_multimap_dictionary(info,Acq_time,FA,Tlist,T2_prep,TI); provenance.cache_status='miss'; provenance_saved=provenance; save(cache_file,'dictionary','provenance_saved','-v7');
end
