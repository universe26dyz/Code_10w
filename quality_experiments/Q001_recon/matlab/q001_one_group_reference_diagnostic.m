function q001_one_group_reference_diagnostic(source_mat, output_mat)
arguments, source_mat (1,:) char, output_mat (1,:) char, end
assert(isfile(source_mat) && ~isfile(output_mat), 'Q001:ReferenceDiagnostic', 'Source must exist and output must be new.');
S=load(source_mat,'Mag_crop','Acq_time','FA','TI','T2_prep','Info_by_slice','final_data_semantics');
assert(strcmp(char(S.final_data_semantics),'MP-PCA(full-FOV MIND_mag_reg)'), 'Q001:ReferenceDiagnostic', 'Formal full-FOV semantics required.');
root=fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(fileparts(root),'MultiMapCode_hhz','Utility'));
[Tlist,dictionary_grid]=q001_formal_dictionary_grid(); total=tic; [t1_ms,t2_ms]=function_T1T2_10HB_bssfp(S.Mag_crop(:,:,:,1),S.Info_by_slice{1},S.Acq_time(:,1),S.FA,Tlist,S.T2_prep,S.TI); total_wall_seconds=toc(total); output_shape=size(t1_ms); source_mat_sha256='recorded_in_q001_input_manifest'; save(output_mat,'t1_ms','t2_ms','total_wall_seconds','output_shape','source_mat_sha256','dictionary_grid','-v7');
end

function [Tlist,grid]=q001_formal_dictionary_grid()
T1_list=[20:20:500,505:5:1500,1520:20:2500]; T2_list=[5:5:100,110:10:200]; B1_list=0.1:0.05:1.2; Tlist=zeros(numel(T1_list)*numel(T2_list)*numel(B1_list),5); n=1;
for i=1:numel(T1_list), for j=1:numel(T2_list), for k=1:numel(B1_list), if T1_list(i)>T2_list(j), Tlist(n,1:3)=[T1_list(i),T2_list(j),B1_list(k)]; n=n+1; end, end, end, end
Tlist(all(Tlist==0,2),:)=[]; Tlist=unique(Tlist,'rows'); grid=struct('num_T1',numel(T1_list),'num_T2',numel(T2_list),'num_B1',numel(B1_list),'num_valid_dictionary_entries',size(Tlist,1)); assert(grid.num_T1==275 && grid.num_T2==30 && grid.num_B1==23 && grid.num_valid_dictionary_entries==186990, 'Q001:ReferenceDiagnosticGrid', 'Diagnostic grid diverges from build_native_reference_maps.m.');
end
