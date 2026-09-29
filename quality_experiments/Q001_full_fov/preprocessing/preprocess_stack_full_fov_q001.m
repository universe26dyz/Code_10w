function preprocess_stack_full_fov_q001(input_dicom_dir, output_mat_path, opts)
%PREPROCESS_STACK_FULL_FOV_Q001 Independent Q001 full-FOV MIND/MP-PCA route.
% This never calls or changes preprocess_stack_v1. It reuses its registered
% MIND and MP-PCA algorithms but deliberately retains every acquired row/col.

arguments
    input_dicom_dir (1,:) char
    output_mat_path (1,:) char
    opts (1,1) struct
end
required = {'max_slices','enable_figures','enable_mppca','mind_alpha','mppca_options'};
for k = 1:numel(required)
    assert(isfield(opts, required{k}), 'Q001:MissingOption', 'opts.%s is required.', required{k});
end
assert(isfolder(input_dicom_dir), 'Q001:InputDirectory', 'Missing DICOM directory: %s', input_dicom_dir);
assert(~isfile(output_mat_path), 'Q001:OutputExists', 'Refusing to overwrite MAT: %s', output_mat_path);
assert(islogical(opts.enable_figures) && isscalar(opts.enable_figures), 'Q001:Figures', 'enable_figures must be logical scalar.');
assert(islogical(opts.enable_mppca) && isscalar(opts.enable_mppca), 'Q001:MPPCA', 'enable_mppca must be logical scalar.');
assert(isnumeric(opts.mind_alpha) && isscalar(opts.mind_alpha) && isfinite(opts.mind_alpha) && opts.mind_alpha > 0, 'Q001:MINDAlpha', 'mind_alpha must be positive finite.');
assert(isstruct(opts.mppca_options), 'Q001:MPPCAOptions', 'mppca_options must be explicit.');
for k = 1:numel({'patch_size','center_data','clip_negative','use_parallel','mask_threshold_fraction','min_component_size','fill_holes','save_residual','overwrite'})
    names = {'patch_size','center_data','clip_negative','use_parallel','mask_threshold_fraction','min_component_size','fill_holes','save_residual','overwrite'};
    assert(isfield(opts.mppca_options, names{k}), 'Q001:MPPCAOptions', 'Missing mppca option %s.', names{k});
end
if isempty(opts.max_slices)
    max_slices = [];
else
    assert(isscalar(opts.max_slices) && isfinite(opts.max_slices) && opts.max_slices >= 1 && opts.max_slices == floor(opts.max_slices), 'Q001:MaxSlices', 'max_slices must be [] or positive integer.');
    max_slices = opts.max_slices;
end

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
baseline_dir = fullfile(repo_root, 'trad', 'modules', 'module_01_preprocess_matlab', 'preprocessing_v1');
addpath(baseline_dir);
dicoms = dir(fullfile(input_dicom_dir, '*.dcm'));
assert(~isempty(dicoms), 'Q001:NoDicom', 'No .dcm files in %s', input_dicom_dir);
n_files = numel(dicoms); raw = []; times = zeros(1,n_files); infos = cell(1,n_files); paths = cell(1,n_files); bases = cell(1,n_files); sops = cell(1,n_files); series = cell(1,n_files);
for k = 1:n_files
    paths{k} = fullfile(dicoms(k).folder, dicoms(k).name); infos{k} = dicominfo(paths{k});
    assert(isfield(infos{k}, 'AcquisitionTime'), 'Q001:AcquisitionTime', 'Missing AcquisitionTime: %s', paths{k});
    assert(isfield(infos{k}, 'SOPInstanceUID') && ~isempty(infos{k}.SOPInstanceUID), 'Q001:SOP', 'Missing SOPInstanceUID: %s', paths{k});
    assert(isfield(infos{k}, 'SeriesInstanceUID') && ~isempty(infos{k}.SeriesInstanceUID), 'Q001:Series', 'Missing SeriesInstanceUID: %s', paths{k});
    image = double(dicomread(paths{k}));
    if k == 1, raw = zeros([size(image), n_files]); else, assert(isequal(size(image), size(raw(:,:,1))), 'Q001:ImageSize', 'Inconsistent image matrix: %s', paths{k}); end
    raw(:,:,k) = image; times(k) = HHMMSS2Sec(infos{k}.AcquisitionTime) * 1e3; bases{k} = dicoms(k).name; sops{k} = char(infos{k}.SOPInstanceUID); series{k} = char(infos{k}.SeriesInstanceUID);
end
[times, order] = sort(times, 'ascend'); raw = raw(:,:,order); infos = infos(order); paths = paths(order); bases = bases(order); sops = sops(order); series = series(order);
NumImg = 10; assert(mod(n_files,NumImg) == 0, 'Q001:Groups', 'Expected multiple of 10 DICOMs, got %d.', n_files);
[nx,ny,~] = size(raw); n_available = n_files / NumImg; Mag = reshape(raw,[nx ny NumImg n_available]); Acq_time = reshape(times,[NumImg n_available]);
if isempty(max_slices), n_slices = n_available; else
    assert(max_slices <= n_available, 'Q001:MaxSlicesRange', 'max_slices exceeds available groups.'); n_slices = max_slices;
    Mag = Mag(:,:,:,1:n_slices); Acq_time = Acq_time(:,1:n_slices); infos = infos(1:NumImg*n_slices); paths = paths(1:NumImg*n_slices); bases = bases(1:NumImg*n_slices); sops = sops(1:NumImg*n_slices); series = series(1:NumImg*n_slices);
end
q001_validate_protocol(infos, paths); q001_validate_group_geometry(infos, paths, NumImg); assert(numel(unique(series)) == 1, 'Q001:Series', 'All DICOMs must share one SeriesInstanceUID.');

% No crop or divisibility condition: MIND and MP-PCA receive the full matrix.
MIND_mag_reg = zeros(size(Mag), 'like', Mag); MIND_mag_reg(:,:,1,:) = Mag(:,:,1,:);
for slice_idx = 1:n_slices
    fixed = single(Mag(:,:,1,slice_idx));
    for weight_idx = 2:NumImg
        [~,~,registered] = deformableReg2Dmind_asym_nodisplay(fixed, single(Mag(:,:,weight_idx,slice_idx)), opts.mind_alpha);
        MIND_mag_reg(:,:,weight_idx,slice_idx) = registered;
    end
end
if opts.enable_mppca
    [Mag_crop, MPPCA_sigma_map, MPPCA_rank_map, MPPCA_support_mask, MPPCA_residual, MPPCA_report] = mppca_denoise_multimap_stack(MIND_mag_reg, opts.mppca_options);
else
    Mag_crop = MIND_mag_reg; MPPCA_sigma_map=[]; MPPCA_rank_map=[]; MPPCA_support_mask=[]; MPPCA_residual=[]; MPPCA_report=[];
end
Info = infos{1}; Info_by_slice = cell(1,n_slices); HB1_source_file_per_slice = cell(1,n_slices);
for slice_idx = 1:n_slices
    source_idx = 1+(slice_idx-1)*NumImg; Info_by_slice{slice_idx}=infos{source_idx}; HB1_source_file_per_slice{slice_idx}=paths{source_idx};
end
TR=Info.RepetitionTime; VPS=Info.EchoTrainLength; FA=[45 45 45]; TI=[50 150]; T2_prep=[35 45 55]; protocol_version='hhz_v1';
spatial_mode='full_fov'; spatial_mode_ascii=uint8(spatial_mode); full_fov_shape_rows_cols=[nx ny]; crop_row_start_zero_based=0; crop_col_start_zero_based=0; crop_size_rows_cols=[nx ny]; crop_indices=struct('rows',1:nx,'cols',1:ny);
if opts.enable_mppca, final_data_semantics='MP-PCA(full-FOV MIND_mag_reg)'; else, final_data_semantics='MIND_mag_reg'; end
final_data_semantics_ascii=uint8(final_data_semantics); geometry_rule='After full-FOV MIND, all 10 weights in each group use HB1 geometry.'; geometry_rule_ascii=uint8(geometry_rule); preprocessing_options=opts; mppca_center_data=opts.mppca_options.center_data;
sop_instance_uid_ascii=q001_ascii(sops); sop_instance_uid_lengths=cellfun(@strlength,string(sops)); series_instance_uid_ascii=q001_ascii(series); series_instance_uid_lengths=cellfun(@strlength,string(series)); source_basename_ascii=q001_ascii(bases); source_basename_lengths=cellfun(@strlength,string(bases)); HB1_sop_instance_uid_ascii=sop_instance_uid_ascii(:,1:NumImg:end); HB1_sop_instance_uid_lengths=sop_instance_uid_lengths(1:NumImg:end);
if opts.enable_figures, figure('Color','w'); imshow(Mag_crop(:,:,1,1),[]); title('Q001 full-FOV HB1'); end
output_dir=fileparts(output_mat_path); assert(isfolder(output_dir), 'Q001:OutputDirectory', 'Output directory must exist: %s', output_dir);
save(output_mat_path,'Mag','Mag_crop','MIND_mag_reg','Acq_time','TR','VPS','FA','TI','T2_prep','Info','Info_by_slice','HB1_source_file_per_slice','paths','crop_indices','crop_row_start_zero_based','crop_col_start_zero_based','crop_size_rows_cols','spatial_mode','spatial_mode_ascii','full_fov_shape_rows_cols','final_data_semantics','final_data_semantics_ascii','geometry_rule','geometry_rule_ascii','protocol_version','preprocessing_options','MPPCA_sigma_map','MPPCA_rank_map','MPPCA_support_mask','MPPCA_residual','MPPCA_report','mppca_center_data','sop_instance_uid_ascii','sop_instance_uid_lengths','series_instance_uid_ascii','series_instance_uid_lengths','source_basename_ascii','source_basename_lengths','HB1_sop_instance_uid_ascii','HB1_sop_instance_uid_lengths','-v7.3');
end

function q001_validate_protocol(infos, paths)
for field = {'RepetitionTime','EchoTrainLength'}
    values=zeros(1,numel(infos));
    for k=1:numel(infos), assert(isfield(infos{k},field{1}) && ~isempty(infos{k}.(field{1})), 'Q001:Protocol', 'Missing %s: %s', field{1},paths{k}); values(k)=double(infos{k}.(field{1})); end
    assert(max(abs(values-values(1))) <= 1e-6*max(1,max(abs(values))), 'Q001:Protocol', '%s is inconsistent.',field{1});
end
end

function q001_validate_group_geometry(infos, paths, num_img)
for group=1:(numel(infos)/num_img)
    index=(group-1)*num_img+(1:num_img); ref=infos{index(1)};
    for name={'ImagePositionPatient','ImageOrientationPatient','PixelSpacing','Rows','Columns','SliceThickness'}
        assert(isfield(ref,name{1}) && ~isempty(ref.(name{1})), 'Q001:Geometry', 'HB1 missing %s.',name{1}); refv=double(ref.(name{1}));
        for k=index
            assert(isfield(infos{k},name{1}) && ~isempty(infos{k}.(name{1})), 'Q001:Geometry', 'Missing %s: %s',name{1},paths{k});
            assert(isequal(size(double(infos{k}.(name{1}))),size(refv)) && all(abs(double(infos{k}.(name{1}))(:)-refv(:)) <= 1e-6*max(1,max(abs(refv(:))))), 'Q001:Geometry', 'Group geometry differs from HB1.');
        end
    end
end
end

function matrix=q001_ascii(values)
width=max(cellfun(@numel,values)); matrix=zeros(width,numel(values),'uint8');
for k=1:numel(values), matrix(1:numel(values{k}),k)=uint8(values{k}); end
end
