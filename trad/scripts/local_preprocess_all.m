function local_preprocess_all(manifest_path, preprocessed_root)
%LOCAL_PREPROCESS_ALL Run local MATLAB preprocessing for only manifest entries.

    arguments
        manifest_path (1, :) char
        preprocessed_root (1, :) char
    end

    assert(isfile(manifest_path), 'local_preprocess_all:Manifest', 'Manifest does not exist: %s', manifest_path);
    assert(isfolder(preprocessed_root), 'local_preprocess_all:OutputRoot', 'Preprocessed root does not exist: %s', preprocessed_root);
    manifest = jsondecode(fileread(manifest_path));
    assert(isfield(manifest, 'schema_version') && strcmp(manifest.schema_version, 'deployment-v1'), ...
        'local_preprocess_all:Schema', 'Manifest must use schema_version deployment-v1.');
    assert(isfield(manifest, 'subjects') && ~isempty(manifest.subjects), ...
        'local_preprocess_all:Subjects', 'Manifest must contain subjects.');
    script_dir = fileparts(mfilename('fullpath'));
    addpath(script_dir);
    seen = strings(0, 1);
    for subject_index = 1:numel(manifest.subjects)
        subject = manifest.subjects(subject_index);
        assert(isfield(subject, 'subject_id') && isfield(subject, 'stacks'), ...
            'local_preprocess_all:Subject', 'Each subject must include subject_id and stacks.');
        for stack_index = 1:numel(subject.stacks)
            entry = subject.stacks(stack_index);
            assert(isfield(entry, 'stack') && isfield(entry, 'dicom_dir'), ...
                'local_preprocess_all:Stack', 'Each stack must include stack and dicom_dir.');
            stack_name = char(entry.stack);
            assert(ismember(stack_name, {'sax', '2ch', '4ch'}), ...
                'local_preprocess_all:StackName', 'Invalid stack: %s', stack_name);
            key = string(subject.subject_id) + "/" + string(stack_name);
            assert(~any(seen == key), 'local_preprocess_all:Duplicate', 'Duplicate subject/stack: %s', key);
            seen(end + 1, 1) = key; %#ok<AGROW>
            output_dir = fullfile(preprocessed_root, char(subject.subject_id), stack_name);
            assert(isfolder(output_dir), 'local_preprocess_all:OutputDirectory', ...
                'Create explicit output directory before preprocessing: %s', output_dir);
            local_preprocess_one(char(entry.dicom_dir), fullfile(output_dir, 'preprocessed.mat'), char(subject.subject_id), stack_name);
        end
    end
end
