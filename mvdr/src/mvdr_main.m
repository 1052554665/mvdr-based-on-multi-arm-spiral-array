%% ============================================================
%% MVDR Beamforming with Real Signals based on RIR
%% Multi-arm Spiral Array
%% ============================================================
%% Main script that orchestrates the entire MVDR pipeline
%% 
%% This script:
%% 1. Loads configuration
%% 2. Loads microphone positions and audio data
%% 3. Generates room impulse responses (RIR)
%% 4. Performs MVDR beamforming (dual-branch: target + interference)
%% 5. Computes diagnostics and generates visualizations
%% ============================================================

clear; close all; clc;

%% Add paths
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
external_rir_dir = fullfile(fileparts(project_root), 'RIR-Generator');

cd(script_dir);  % Change to script directory first so relative paths are stable
addpath(genpath(project_root));  % Project source tree
if exist(external_rir_dir, 'dir') == 7
    addpath(genpath(external_rir_dir));  % External RIR generator dependency
else
    warning('MVDR:Path', 'RIR-Generator folder not found at %s', external_rir_dir);
end

fprintf('====================================================\n');
fprintf('  MVDR Beamforming with Real Signals (RIR-based)\n');
fprintf('====================================================\n\n');

%% 1. Load configuration
fprintf('[Main] Loading configuration...\n');
experiment_name = 'real_signal_dcbias_4k';
% experiment_name = 'demo_2k_4k';
config = load_config(experiment_name);  % Swap this to run a different experiment override

%% 2. Create output directories if needed
if ~isfolder(config.output_dir)
    mkdir(config.output_dir);
end

if config.save_figures && ~isfolder(fullfile(config.output_dir, config.figure_subdir))
    mkdir(fullfile(config.output_dir, config.figure_subdir));
end

if ~isfolder(fullfile(config.output_dir, 'results'))
    mkdir(fullfile(config.output_dir, 'results'));
end

if config.save_figures
    fprintf('[Main] Output directories ready.\n');
end

%% 3. Load microphone positions and audio signals
[mic_pos, X_target, X_interf, X_noisy, x_target, x_interf, t, Nt, Nmic, r_center] = ...
    load_data_module(config);

%% 4. Generate RIRs and convolve with signals
[h_target, h_interf, X_target, X_interf, X_noisy] = ...
    rir_generation_module(config, x_target, x_interf, mic_pos, Nt, Nmic);

%% 5. MVDR processing (beamforming + STFTs)
[Yf_tgt, Y_tar_tgt, Y_intnoi_tgt, Yf_int, Y_interf_int, Y_tarnoi_int, ...
 S, S_tar, S_interf, S_intnoi, S_tarnoi, F, T, a_target, a_interf] = ...
    mvdr_processing_module(config, X_noisy, X_target, X_interf, mic_pos, r_center, Nt, Nmic);

%% 6. Diagnostics and visualization
diagnostics_and_visualization_module(config, Yf_tgt, Y_tar_tgt, Y_intnoi_tgt, ...
    Yf_int, Y_interf_int, Y_tarnoi_int, S, S_interf, S_intnoi, S_tarnoi, ...
    X_noisy, X_target, X_interf, F, T, a_target, a_interf, mic_pos, r_center, Nt, Nmic);

fprintf('\n====================================================\n');
fprintf('  Pipeline completed successfully!\n');
fprintf('  Check %s/figures/ for visualizations\n', config.output_dir);
fprintf('====================================================\n');
