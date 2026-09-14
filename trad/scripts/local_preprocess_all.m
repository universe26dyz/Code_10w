function local_preprocess_all(manifest_path, preprocessed_root)
%LOCAL_PREPROCESS_ALL Run local MATLAB preprocessing for only manifest entries.

    arguments
        manifest_path (1, :) char
        preprocessed_root (1, :) char
    end

    assert(isfile(manifest_path), 'local_preprocess_all:Manifest', 'Manifest does not exist: %s', manifest_path);
    if ~isfolder(preprocessed_root)
        [ok, message] = mkdir(preprocessed_root);
        assert(ok, 'local_preprocess_all:OutputRoot', 'Cannot create preprocessed root %s: %s', preprocessed_root, message);
    end
    manifest = jsondecode(fileread(manifest_path));
    assert(isfield(manifest, 'schema_version') && strcmp(manifest.schema_version, 'deployment-v1.1'), ...
        'local_preprocess_all:Schema', 'Manifest must use deployment-v1.1; migrate old ambiguous dicom_dir entries.');
    assert(isfield(manifest, 'subjects') && ~isempty(manifest.subjects), ...
        'local_preprocess_all:Subjects', 'Manifest must contain subjects.');
    script_dir = fileparts(mfilename('fullpath'));
    addpath(script_dir);
    seen = strings(0, 1);
    seen_subjects = strings(0, 1);
    for subject_index = 1:numel(manifest.subjects)
        subject = manifest.subjects(subject_index);
        assert(isfield(subject, 'subject_id') && isfield(subject, 'stacks'), ...
            'local_preprocess_all:Subject', 'Each subject must include subject_id and stacks.');
        subject_id = char(subject.subject_id);
        assert(~isempty(strtrim(subject_id)), 'local_preprocess_all:SubjectID', 'subject_id must be non-empty.');
        assert(~any(seen_subjects == string(subject_id)), 'local_preprocess_all:SubjectID', 'Duplicate subject_id: %s', subject_id);
        seen_subjects(end + 1, 1) = string(subject_id); %#ok<AGROW>
        for stack_index = 1:numel(subject.stacks)
            entry = subject.stacks(stack_index);
            assert(isfield(entry, 'stack') && isfield(entry, 'local_dicom_dir') && isfield(entry, 'server_dicom_dir'), ...
                'local_preprocess_all:Stack', 'Each stack must include stack, local_dicom_dir and server_dicom_dir.');
            stack_name = char(entry.stack);
            assert(ismember(stack_name, {'sax', '2ch', '4ch'}), ...
                'local_preprocess_all:StackName', 'Invalid stack: %s', stack_name);
            local_dicom_dir = char(entry.local_dicom_dir);
            assert(~isempty(strtrim(local_dicom_dir)), 'local_preprocess_all:LocalDicom', 'local_dicom_dir must be non-empty.');
            assert(~isempty(strtrim(char(entry.server_dicom_dir))), 'local_preprocess_all:ServerDicom', 'server_dicom_dir must be non-empty.');
            key = string(subject_id) + "/" + string(stack_name);
            assert(~any(seen == key), 'local_preprocess_all:Duplicate', 'Duplicate subject/stack: %s', key);
            seen(end + 1, 1) = key; %#ok<AGROW>
            output_dir = fullfile(preprocessed_root, subject_id, stack_name);
            if ~isfolder(output_dir)
                [ok, message] = mkdir(output_dir);
                assert(ok, 'local_preprocess_all:OutputDirectory', 'Cannot create output directory %s: %s', output_dir, message);
            end
            local_preprocess_one(local_dicom_dir, fullfile(output_dir, 'preprocessed.mat'), subject_id, stack_name);
        end
    end
end
