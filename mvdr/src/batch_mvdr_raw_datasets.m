function batch_mvdr_raw_datasets(raw_root, experiment_name, output_root)
%% ============================================================
%% Batch MVDR processing for raw_datasets
%% ============================================================
% Scans a dataset tree where Normal is the interference class and all
% other subfolders are target classes. Each target file is randomly mixed
% with one Normal file, then the existing MVDR pipeline is executed.

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
external_rir_dir = fullfile(fileparts(project_root), 'RIR-Generator');

cd(script_dir);
addpath(genpath(project_root));
if exist(external_rir_dir, 'dir') == 7
    addpath(genpath(external_rir_dir));
else
    warning('MVDR:Path', 'RIR-Generator folder not found at %s', external_rir_dir);
end

if nargin < 1 || isempty(raw_root)
    raw_root = fullfile(fileparts(project_root), 'raw_datasets');
end
if nargin < 2 || isempty(experiment_name)
    experiment_name = 'real_signal_dcbias_4k';
end
if nargin < 3 || isempty(output_root)
    output_root = fullfile(project_root, 'output', 'raw_datasets_mvdr');
end

base_config = load_config(experiment_name);
base_config.save_figures = true;
base_config.figure_subdir = 'figures';
base_config.output_dir = output_root;

if ~isfolder(output_root)
    mkdir(output_root);
end

split_dirs = discover_split_dirs(raw_root);
if isempty(split_dirs)
    error('No dataset splits found under %s. Expected train/val/test or class folders directly under raw_root.', raw_root);
end

rng(42, 'twister');
manifest = table();

fprintf('\n====================================================\n');
fprintf('  Batch MVDR on raw_datasets\n');
fprintf('====================================================\n\n');

for s = 1:numel(split_dirs)
    split_name = split_dirs{s}.name;
    split_root = split_dirs{s}.path;
    class_dirs = discover_class_dirs(split_root);

    normal_dir = find_normal_dir(class_dirs);
    if isempty(normal_dir)
        warning('Split "%s" has no Normal folder. Skipping.', split_name);
        continue;
    end

    normal_files = collect_audio_files(normal_dir);
    if isempty(normal_files)
        warning('Split "%s" has no audio files in Normal. Skipping.', split_name);
        continue;
    end

    for c = 1:numel(class_dirs)
        class_name = class_dirs(c).name;
        class_path = class_dirs(c).path;

        if is_normal_name(class_name)
            continue;
        end

        target_files = collect_audio_files(class_path);
        if isempty(target_files)
            warning('Class "%s" has no audio files. Skipping.', class_name);
            continue;
        end

        fprintf('[Batch] Split=%s | Target class=%s | %d files\n', split_name, class_name, numel(target_files));

        for i = 1:numel(target_files)
            target_file = target_files{i};
            interf_file = normal_files{randi(numel(normal_files))};

            [~, target_base, ~] = fileparts(target_file);
            [~, interf_base, ~] = fileparts(interf_file);
            sample_tag = sprintf('%s__%s', sanitize_name(target_base), sanitize_name(interf_base));
            sample_out_dir = fullfile(output_root, split_name, class_name, sample_tag);
            ensure_output_dirs(sample_out_dir);

            config = base_config;
            config.experiment_name = sprintf('%s_%s_%s', experiment_name, split_name, sanitize_name(class_name));
            config.target_audio = target_file;
            config.interf_audio = interf_file;
            config.output_dir = sample_out_dir;
            config.figure_subdir = 'figures';

            fprintf('[Batch]   (%d/%d) target=%s | interf=%s\n', i, numel(target_files), target_base, interf_base);

            try
                [mic_pos, X_target, X_interf, X_noisy, x_target, x_interf, t, Nt, Nmic, r_center] = load_data_module(config);
                [~, X_target, X_interf, X_noisy] = rir_generation_module(config, x_target, x_interf, mic_pos, Nt, Nmic);
                [Yf_tgt, Y_tar_tgt, Y_intnoi_tgt, Yf_int, Y_interf_int, Y_tarnoi_int, S, S_tar, S_interf, S_intnoi, S_tarnoi, F, T, a_target, a_interf] = ...
                    mvdr_processing_module(config, X_noisy, X_target, X_interf, mic_pos, r_center, Nt, Nmic);
                diagnostics_and_visualization_module(config, Yf_tgt, Y_tar_tgt, Y_intnoi_tgt, ...
                    Yf_int, Y_interf_int, Y_tarnoi_int, S, S_interf, S_intnoi, S_tarnoi, ...
                    X_noisy, X_target, X_interf, F, T, a_target, a_interf, mic_pos, r_center, Nt, Nmic);

                manifest = [manifest; make_manifest_row(split_name, class_name, target_file, interf_file, sample_out_dir, 'ok')]; %#ok<AGROW>
            catch exc
                warning('Failed on split=%s class=%s target=%s: %s', split_name, class_name, target_base, exc.message);
                manifest = [manifest; make_manifest_row(split_name, class_name, target_file, interf_file, sample_out_dir, ['failed: ' exc.message])]; %#ok<AGROW>
            end
        end
    end
end

if ~isempty(manifest)
    manifest_path = fullfile(output_root, 'batch_manifest.csv');
    writetable(manifest, manifest_path);
    fprintf('\n[Batch] Manifest saved to %s\n', manifest_path);
end

fprintf('\n[Batch] Completed. Outputs are under %s\n', output_root);

end

function split_dirs = discover_split_dirs(raw_root)
split_dirs = struct('name', {}, 'path', {});
root_entries = dir(raw_root);
root_entries = root_entries([root_entries.isdir]);
root_entries = root_entries(~ismember({root_entries.name}, {'.', '..'}));

known_splits = {'train', 'val', 'test'};
found_known = false;
for k = 1:numel(root_entries)
    entry_name = root_entries(k).name;
    if any(strcmpi(entry_name, known_splits))
        found_known = true;
        split_dirs(end + 1).name = entry_name; %#ok<AGROW>
        split_dirs(end).path = fullfile(raw_root, entry_name);
    end
end

if ~found_known
    split_dirs(1).name = 'all';
    split_dirs(1).path = raw_root;
end
end

function class_dirs = discover_class_dirs(split_root)
entries = dir(split_root);
entries = entries([entries.isdir]);
entries = entries(~ismember({entries.name}, {'.', '..'}));
class_dirs = struct('name', {}, 'path', {});
for k = 1:numel(entries)
    class_dirs(end + 1).name = entries(k).name; %#ok<AGROW>
    class_dirs(end).path = fullfile(split_root, entries(k).name);
end
end

function normal_dir = find_normal_dir(class_dirs)
normal_dir = '';
for k = 1:numel(class_dirs)
    if is_normal_name(class_dirs(k).name)
        normal_dir = class_dirs(k).path;
        return;
    end
end
end

function tf = is_normal_name(name)
tf = strcmpi(name, 'Normal') || strcmpi(name, 'normal');
end

function files = collect_audio_files(folder)
extensions = {'.wav', '.flac', '.mp3', '.ogg'};
entries = dir(folder);
files = {};
for k = 1:numel(entries)
    if entries(k).isdir
        continue;
    end
    [~, ~, ext] = fileparts(entries(k).name);
    if any(strcmpi(ext, extensions))
        files{end + 1} = fullfile(folder, entries(k).name); %#ok<AGROW>
    end
end
end

function ensure_output_dirs(sample_out_dir)
if ~isfolder(sample_out_dir)
    mkdir(sample_out_dir);
end
figures_dir = fullfile(sample_out_dir, 'figures');
results_dir = fullfile(sample_out_dir, 'results');
if ~isfolder(figures_dir)
    mkdir(figures_dir);
end
if ~isfolder(results_dir)
    mkdir(results_dir);
end
end

function name = sanitize_name(name)
name = regexprep(name, '[^a-zA-Z0-9_\-]', '_');
end

function row = make_manifest_row(split_name, class_name, target_file, interf_file, output_dir, status)
row = table(string(split_name), string(class_name), string(target_file), string(interf_file), string(output_dir), string(status), ...
    'VariableNames', {'split', 'target_class', 'target_file', 'interf_file', 'output_dir', 'status'});
end