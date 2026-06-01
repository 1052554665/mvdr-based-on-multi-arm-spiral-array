function config = mvdr_default_config()
%% ============== MVDR Default Configuration ==============
% Shared baseline parameters for all MVDR experiments.

%% Audio and signal parameters
config.c = 340;                    % Speed of sound (m/s)
config.fs = 16000;                 % Sampling frequency (Hz)
config.SNR = 30;                   % SNR target (dB)
config.nsample = 4096;             % RIR length (samples)

%% Source positions (3D coordinates in meters)
config.s_target = [2.5 1 4];       % Target source position
config.s_interf = [2.5 3 4];       % Interference source position

%% Array center
config.array_center = [2.5 2 1.5]; % Array center (m)

%% Room parameters
config.room_L = [5 4 6];           % Room dimensions [length, width, height] (m)
config.beta = 0;                   % Wall reflection coefficient (0=direct only)
config.mtype = 'omnidirectional';  % Microphone type
config.order = -1;                 % RIR reflection order (-1=all)
config.dim = 3;                    % Dimension (3 for 3D)
config.orientation = 0;            % Microphone orientation
config.hp_filter = true;           % High-pass filter flag

%% MVDR parameters
config.mvdr_fmin_hz = 300;         % MVDR minimum frequency (Hz)
config.mvdr_fmax_hz = 8000;        % MVDR maximum frequency (Hz)
config.mvdr_use_time_varying = false;  % Time-varying weights (true=slower, more adaptive)
config.mvdr_frame_stride = 2;      % Update weights every N frames
config.mvdr_progress_step = 20;    % Print progress every N bins

%% Covariance estimation parameters
config.Mavg = 31;                  % Averaging window for covariance (frames)
config.epsilon = 1e-3;             % Base diagonal loading (relative scale)
config.shrink_alpha = 0;        % Shrinkage factor (0..0.3)
config.M_signal = 2;               % Signal subspace dimension (0=full rank)

%% Oracle covariance modes (for debug/upper bound evaluation)
config.use_oracle_intnoi_cov = true;   % Use true interference+noise cov (target-steered)
config.use_oracle_tarnoi_cov = true;    % Use true target+noise cov (interference-steered)

%% Source signal normalization
config.normalize_source_rms = true;    % Normalize source RMS
config.INR_dB = 0;                     % Interference-to-target level before room (dB)

%% STFT parameters
config.Nfft = 1024;                % FFT length
config.win_len = 512;              % Window length
config.noverlap = 256;             % Overlap length

%% GPU acceleration (requires Parallel Computing Toolbox)
config.use_gpu = true;
config.use_parallel_rir = true;    % Parallelize RIR generation

%% Reference microphone for metrics
config.ref_mic = 1;

%% Array geometry parameters (Fraunhofer check)
config.array_diameter = 0.15;      % Maximum array diameter (m)

%% Display and output
config.verbose = true;             % Verbose console output
config.save_figures = true;        % Save PDF figures
config.output_dir = '../output';   % Output directory for figures
config.figure_subdir = 'real_signal'; % Subdirectory for figures/results inside output_dir

config.plot_style = struct();
config.plot_style.figure_bg = 'white';
config.plot_style.axes_bg = 'white';
config.plot_style.text_color = 'k';
config.plot_style.colormap_name = 'turbo';
config.plot_style.use_invert_hardcopy = false;
config.plot_style.line_width = struct('main', 1.5, 'medium', 2.0, 'thick', 2.5, 'thin', 1.2);
config.plot_style.marker_size = struct('main', 8, 'detail', 10);
config.plot_style.font_sz = struct(...
	'title', 32, ...
	'subtitle', 31, ...
	'label', 31, ...
	'subtile_title', 31, ...
	'legend', 30, ...
	'tick', 30, ...
	'small_tick', 29, ...
	'colorbar', 30, ...
	'colorbar_label', 30);

config.legend_mvdr = {'Target-steered', 'Interference-steered'};
config.legend_psd = {'Target-steered', 'Interference-steered'};
config.eigen_plot_names = {'2kHz only', 'Full-band (2k+4k)'};

config.figure213_annotations = struct();
config.figure213_annotations.mixed = struct( ...
	'text', '2kHz+4kHz', ...
	'font_size', 15, ...
	'color', 'w', ...
	'font_weight', 'bold', ...
	'background_color', [0 0 0 0.5]);
config.figure213_annotations.target = struct( ...
	'text', 'Suppress 4kHz', ...
	'font_size', 15, ...
	'color', 'w', ...
	'font_weight', 'bold', ...
	'background_color', [0 0.5 0 0.5]);
config.figure213_annotations.interference = struct( ...
	'text', 'Suppress 2kHz', ...
	'font_size', 15, ...
	'color', 'w', ...
	'font_weight', 'bold', ...
	'background_color', [0.5 0 0 0.5]);

%% File paths (relative to src/ directory)
config.data_dir = '../data';
config.mic_pos_file = '../data/mic_positions.xlsx';
config.target_audio = '../data/audio/DCBias2_60.wav';
config.interf_audio = '../data/audio/sine_wave_4k.wav';

config.experiment_name = 'real_signal_dcbias_4k';

end
