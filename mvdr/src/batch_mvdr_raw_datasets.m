function batch_mvdr_raw_datasets(raw_root, experiment_name, output_root)
%% ============================================================
%% Batch MVDR processing for raw_datasets
%% ============================================================
% Pairs one representative waveform from each non-Normal class with
% a Normal interference waveform. Each pair is processed by the MVDR
% pipeline, exporting exactly two axis-free spectrograms per class:
%   1) target_with_interference.png
%   2) interference_with_target.png
%
% Output structure (flat, one folder per class):
%   output/<experiment_name>/
%   ├── DCBias/
%   │   ├── target_with_interference.png
%   │   └── interference_with_target.png
%   ├── Harmonic/
%   ├── Loosen/
%   └── PartialDischarge/

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

% Bust MATLAB M-code cache to ensure edited modules are picked up
clear functions;
fprintf('[Batch] MATLAB function cache cleared.\n');

if nargin < 1 || isempty(raw_root)
    raw_root = fullfile(project_root, 'raw_datasets');
end
if nargin < 2 || isempty(experiment_name)
    experiment_name = 'batch_pairwise';
end
if nargin < 3 || isempty(output_root)
    output_root = fullfile(project_root, 'output', experiment_name);
end

base_config = load_config(experiment_name);
base_config.output_dir = output_root;
base_config.use_parallel_rir = false;

fprintf('[Batch] Experiment: %s\n', experiment_name);
fprintf('[Batch] Output root: %s\n', output_root);

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

fprintf('[Batch] Configuration check:\n');
fprintf('  save_spectrums_only = %d\n', isfield(base_config, 'save_spectrums_only') && base_config.save_spectrums_only);
fprintf('  figure_subdir = %s\n', base_config.figure_subdir);
fprintf('  output_dir = %s\n', base_config.output_dir);
fprintf('  save_figures = %d\n', base_config.save_figures);
fprintf('  win_len = %d\n', base_config.win_len);
fprintf('\n');

for s = 1:numel(split_dirs)
    split_name = split_dirs(s).name;
    split_root = split_dirs(s).path;
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

        fprintf('[Batch] Class=%s | %d target file(s) available\n', class_name, numel(target_files));

        % Pick one representative target + one fixed interference per class
        target_file  = target_files{1};                         % first file in class
        interf_file  = normal_files{1};                         % fixed Normal reference
        [~, target_base, ~] = fileparts(target_file);
        [~, interf_base, ~] = fileparts(interf_file);

        % Flat output: <output_root>/<ClassName>/
        sample_out_dir = fullfile(output_root, class_name);

        spectrum_only_mode = isfield(base_config, 'save_spectrums_only') && base_config.save_spectrums_only;
        ensure_output_dirs(sample_out_dir, spectrum_only_mode);

        config = base_config;
        config.experiment_name = sprintf('%s_%s', experiment_name, sanitize_name(class_name));
        config.target_audio  = target_file;
        config.interf_audio   = interf_file;
        config.output_dir     = sample_out_dir;

        if spectrum_only_mode
            config.figure_subdir = '';            % save PNGs directly in class folder
        end

        fprintf('[Batch]   pair: target=%s | interference=%s\n', target_base, interf_base);
        if spectrum_only_mode
            fprintf('[Batch]   Mode: SPECTRUM_ONLY → 2 PNGs in %s/\n', sample_out_dir);
        else
            fprintf('[Batch]   Mode: FULL_DIAGNOSTICS\n');
        end

        % Pre-flight: check audio file durations
        try
            info_t = audioinfo(target_file);
            info_i = audioinfo(interf_file);
            min_dur_sec = config.win_len / config.fs;
            if info_t.Duration < min_dur_sec || info_i.Duration < min_dur_sec
                skip_reason = sprintf('skipped: audio too short (target=%.0f ms, interf=%.0f ms, min=%.0f ms)', ...
                    info_t.Duration*1000, info_i.Duration*1000, min_dur_sec*1000);
                fprintf('[Batch]   ⚠ SKIPPING: %s\n', skip_reason);
                manifest = [manifest; make_manifest_row(class_name, target_file, interf_file, sample_out_dir, skip_reason)]; %#ok<AGROW>
                continue;
            end
        catch info_exc
            fprintf('[Batch]   ⚠ WARNING: Could not read audio info: %s. Proceeding anyway...\n', info_exc.message);
        end

        try
            fprintf('[Batch]   Step 1/3: Loading data module...\n');
            [mic_pos, ~, ~, ~, x_target, x_interf, ~, Nt, Nmic, r_center] = load_data_module(config);

            fprintf('[Batch]   Step 2/3: Generating RIRs...\n');
            [~, X_target, X_interf, X_noisy] = rir_generation_module(config, x_target, x_interf, mic_pos, Nt, Nmic);

            if spectrum_only_mode
                fprintf('[Batch]   Step 3/3: Running MVDR and saving two spectrograms...\n');

                % Belt-and-suspenders: ensure all signals have consistent ≥ win_len rows
                target_len = max([size(X_noisy,1), size(X_target,1), size(X_interf,1), config.win_len]);
                if size(X_noisy,1)  < target_len, X_noisy(target_len, Nmic)  = 0; end
                if size(X_target,1) < target_len, X_target(target_len, Nmic) = 0; end
                if size(X_interf,1) < target_len, X_interf(target_len, Nmic) = 0; end
                fprintf('[Batch]   Signal dims: X_noisy [%d×%d], X_target [%d×%d], X_interf [%d×%d]\n', ...
                    size(X_noisy,1), size(X_noisy,2), size(X_target,1), size(X_target,2), size(X_interf,1), size(X_interf,2));

                [Yf_tgt, ~, ~, Yf_int, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = ...
                    mvdr_processing_module(config, X_noisy, X_target, X_interf, mic_pos, r_center, Nt, Nmic);

                save_two_spectrograms_only(config, Yf_tgt, Yf_int);
                fprintf('[Batch]   ✓ Spectrograms saved\n');
            else
                [Yf_tgt, Y_tar_tgt, Y_intnoi_tgt, Yf_int, Y_interf_int, Y_tarnoi_int, ...
                    S, S_tar, S_interf, S_intnoi, S_tarnoi, F, T, a_target, a_interf] = ... %#ok<NASGU>
                    mvdr_processing_module(config, X_noisy, X_target, X_interf, mic_pos, r_center, Nt, Nmic);
                diagnostics_and_visualization_module(config, Yf_tgt, Y_tar_tgt, Y_intnoi_tgt, ...
                    Yf_int, Y_interf_int, Y_tarnoi_int, S, S_interf, S_intnoi, S_tarnoi, ...
                    X_noisy, X_target, X_interf, F, T, a_target, a_interf, mic_pos, r_center, Nt, Nmic);
            end

            manifest = [manifest; make_manifest_row(class_name, target_file, interf_file, sample_out_dir, 'ok')]; %#ok<AGROW>
        catch exc
            fprintf('[Batch]   ✗ ERROR: %s\n', exc.message);
            fprintf('[Batch]     in %s line %d\n', exc.stack(1).name, exc.stack(1).line);
            warning('Failed on class=%s target=%s: %s', class_name, target_base, exc.message);
            manifest = [manifest; make_manifest_row(class_name, target_file, interf_file, sample_out_dir, ['failed: ' exc.message])]; %#ok<AGROW>
        end
    end
end

if ~isempty(manifest)
    manifest_path = fullfile(output_root, 'batch_manifest.csv');
    writetable(manifest, manifest_path);
    fprintf('\n[Batch] Manifest saved to %s\n', manifest_path);

    % Print summary
    n_total = height(manifest);
    n_ok = sum(contains(manifest.status, 'ok'));
    n_skip = sum(contains(manifest.status, 'skipped'));
    n_fail = sum(contains(manifest.status, 'failed'));
    fprintf('[Batch] ====== SUMMARY ======\n');
    fprintf('[Batch]   Total pairs:  %d\n', n_total);
    fprintf('[Batch]   OK:           %d\n', n_ok);
    fprintf('[Batch]   Skipped:      %d\n', n_skip);
    fprintf('[Batch]   Failed:       %d\n', n_fail);
    fprintf('[Batch] =====================\n');
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

function class_dir = find_class_dir(class_dirs, target_name)
class_dir = '';
for k = 1:numel(class_dirs)
    if strcmpi(class_dirs(k).name, target_name)
        class_dir = class_dirs(k).path;
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

function ensure_output_dirs(sample_out_dir, spectrum_only_mode)
if ~isfolder(sample_out_dir)
    mkdir(sample_out_dir);
end

if ~spectrum_only_mode
    figures_dir = fullfile(sample_out_dir, 'figures');
    results_dir = fullfile(sample_out_dir, 'results');
    if ~isfolder(figures_dir)
        mkdir(figures_dir);
    end
    if ~isfolder(results_dir)
        mkdir(results_dir);
    end
end
end

function name = sanitize_name(name)
name = regexprep(name, '[^a-zA-Z0-9_\-]', '_');
end

function save_two_spectrograms_only(config, Yf_tgt, Yf_int)
%% Save exactly two MVDR spectrograms into config.output_dir
% When config.figure_subdir is non-empty, saves into that subdirectory;
% otherwise saves directly into config.output_dir.

fprintf('[Batch]     Saving two spectrograms...\n');

if isfield(config, 'figure_subdir') && ~isempty(config.figure_subdir)
    spectrum_dir = fullfile(config.output_dir, config.figure_subdir);
else
    spectrum_dir = config.output_dir;
end
fprintf('[Batch]     Output dir: %s\n', spectrum_dir);

if ~isfolder(spectrum_dir)
    mkdir(spectrum_dir);
end

plot_style = config.plot_style;
if isfield(plot_style, 'colormap_name')
    cmap_name = plot_style.colormap_name;
else
    cmap_name = 'turbo';
end

P_tgt = 20 * log10(abs(Yf_tgt) + eps);
P_int = 20 * log10(abs(Yf_int) + eps);
all_spec_vals = [P_tgt(:); P_int(:)];
spec_clim_max = max(all_spec_vals);
spec_clim_min = spec_clim_max - 80;

fprintf('[Batch]     Color limits: min=%.2f, max=%.2f dB\n', spec_clim_min, spec_clim_max);

save_single_spectrum(spectrum_dir, 'target_with_interference', P_tgt, cmap_name, spec_clim_min, spec_clim_max);
save_single_spectrum(spectrum_dir, 'interference_with_target', P_int, cmap_name, spec_clim_min, spec_clim_max);

fprintf('[Batch]     ✓ 2 PNGs saved to %s/\n', spectrum_dir);
end

function save_single_spectrum(output_dir, base_name, spec_db, cmap_name, spec_min, spec_max)
%% Save a single spectrogram as a 224×224 PNG (transformer-ready).
file_path = fullfile(output_dir, [base_name '.png']);

if spec_max <= spec_min
    spec_max = spec_min + 1;
end

% Resize spectrogram data to 224×224
spec_resized = imresize(spec_db, [224, 224], 'bilinear');

% Map dB values to [0, 1] then to 8-bit colormap indices
spec_norm = (spec_resized - spec_min) / (spec_max - spec_min);
spec_norm = min(max(spec_norm, 0), 1);          % clamp to [0, 1]

cmap = feval(cmap_name, 256);
img_rgb = ind2rgb(uint8(spec_norm * 255), cmap);

imwrite(img_rgb, file_path);
end

function row = make_manifest_row(class_name, target_file, interf_file, output_dir, status)
row = table(string(class_name), string(target_file), string(interf_file), string(output_dir), string(status), ...
    'VariableNames', {'target_class', 'target_file', 'interf_file', 'output_dir', 'status'});
end
