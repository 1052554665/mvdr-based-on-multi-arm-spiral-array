function overrides = mvdr_config_batch_pairwise()
%% ============== Experiment Override: Batch Pairwise MVDR Spectrogram Export ==============
% Batch export preset for raw_datasets.
%
% Pairing logic (handled by batch_mvdr_raw_datasets.m):
%   - target  ← one of {DCBias, Harmonic, Loosen, PartialDischarge}
%   - interference ← Normal
% Each class is processed as a separate experiment; Normal always serves
% as the interference source.
%
% Output per pair: exactly two axis-free spectrograms saved as PNG:
%   1) target_with_interference.png   — MVDR steered toward target
%   2) interference_with_target.png   — MVDR steered toward interference
%
% These spectrograms are suitable for dual-channel transformer training.

overrides = struct();

%% ---- Experiment identity ----
overrides.experiment_name = 'batch_pairwise';

%% ---- Output paths ----
overrides.output_dir      = '../output/batch_pairwise';
overrides.figure_subdir   = '';          % empty → PNGs saved directly in class folder

%% ---- Output control ----
overrides.save_figures          = true;
overrides.save_spectrums_only   = true;    % Only the two MVDR spectrograms
overrides.spectrum_remove_axes  = true;    % Axis-free for transformer input
overrides.spectrum_output_format = 'png';

%% ---- Performance ----
overrides.use_parallel_rir = false;        % Serial RIR generation (more stable in batch)

%% ---- STFT parameters (reduced window for short audio clips) ----
overrides.win_len  = 256;                  % Window length (default 512 → 256)
overrides.noverlap = 128;                  % Overlap length (default 256 → 128)
% Nfft = 1024 is inherited from mvdr_default_config

%% ---- Placeholders (overwritten per sample by batch driver) ----
overrides.target_audio  = '';
overrides.interf_audio   = '';

%% ---- Plot style (publication-quality, white background) ----
overrides.plot_style = struct();
overrides.plot_style.figure_bg       = 'white';
overrides.plot_style.axes_bg         = 'white';
overrides.plot_style.text_color      = 'k';
overrides.plot_style.colormap_name   = 'turbo';
overrides.plot_style.use_invert_hardcopy = false;

overrides.plot_style.line_width = struct(...
    'main', 1.5, 'medium', 2.0, 'thick', 2.5, 'thin', 1.2);

overrides.plot_style.marker_size = struct(...
    'main', 8, 'detail', 10);

overrides.plot_style.font_sz = struct(...
    'title',          30, ...
    'subtitle',       29, ...
    'label',          29, ...
    'subtile_title',  29, ...
    'legend',         28, ...
    'tick',           28, ...
    'small_tick',     27, ...
    'colorbar',       28, ...
    'colorbar_label', 28);

end
