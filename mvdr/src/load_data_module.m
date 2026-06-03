%% ============== Load Data Module ==============
% Loads microphone positions, audio signals, and generates visualization

function [mic_pos, X_target, X_interf, X_noisy, x_target, x_interf, t, Nt, Nmic, r_center] = load_data_module(config)

fprintf('\n[Load Data] Starting data loading...\n');

%% 1. Read microphone positions
fprintf('[Load Data] Reading microphone positions from %s\n', config.mic_pos_file);
data = readmatrix(config.mic_pos_file);
X = data(:,1); 
Y = data(:,2);
Z = zeros(size(X));  % Assume Z column if available, otherwise all zeros

mic_pos = [X Y Z] / 1000;  % Convert mm to m (adjust if cm: use /100)
mic_pos = mic_pos - mean(mic_pos,1) + config.array_center;

Nmic = size(mic_pos,1);
r_center = mean(mic_pos,1);  % Array center for plane-wave phase reference

fprintf('[Load Data] Loaded %d microphones, array center: [%.3f, %.3f, %.3f] m\n', ...
    Nmic, r_center(1), r_center(2), r_center(3));

%% 2. Load audio signals
fprintf('[Load Data] Loading audio signals\n');
[target_sig, fs_t] = audioread(config.target_audio);
[interf_sig, fs_i] = audioread(config.interf_audio);

% Mono conversion
if size(target_sig,2) > 1
    target_sig = mean(target_sig,2);
end
if size(interf_sig,2) > 1
    interf_sig = mean(interf_sig,2);
end

% Resample to system fs
if fs_t ~= config.fs
    fprintf('[Load Data] Resampling target from %d Hz to %d Hz\n', fs_t, config.fs);
    target_sig = resample(target_sig, config.fs, fs_t);
end
if fs_i ~= config.fs
    fprintf('[Load Data] Resampling interference from %d Hz to %d Hz\n', fs_i, config.fs);
    interf_sig = resample(interf_sig, config.fs, fs_i);
end

% Length alignment with padding so short clips still support the STFT window.
% Note: padding happens AFTER RMS normalization so that zero-padding
% does not artificially reduce the RMS of the active portion.
Nt_raw = max(length(target_sig), length(interf_sig));
Nt = max(Nt_raw, config.win_len);

fprintf('[Load Data] After resampling: target=%d samples, interference=%d samples\n', ...
    length(target_sig), length(interf_sig));
fprintf('[Load Data] Nt_raw=%d, Nt=%d (win_len=%d)%s\n', ...
    Nt_raw, Nt, config.win_len, iif(Nt > Nt_raw, ' *** PADDED ***', ''));

x_target = zeros(Nt, 1);
x_interf = zeros(Nt, 1);
x_target(1:length(target_sig)) = target_sig;
x_interf(1:length(interf_sig)) = interf_sig;

% Normalize RMS
if config.normalize_source_rms
    x_target = x_target / (sqrt(mean(x_target.^2)) + eps);
    x_interf = x_interf / (sqrt(mean(x_interf.^2)) + eps);
    fprintf('[Load Data] Source signals normalized to unit RMS\n');
end

% Set INR before room propagation
x_interf = x_interf * 10^(config.INR_dB/20);

t = (0:Nt-1).' / config.fs;
fprintf('[Load Data] Final audio signals: %d samples (%.2f sec) at %d Hz\n', Nt, Nt/config.fs, config.fs);

%% 3. Check far-field criterion (Fraunhofer distance)
fprintf('[Load Data] Checking far-field criterion\n');

f_max = config.mvdr_fmax_hz;
lambda_min = config.c / f_max;
R_far = 2 * config.array_diameter^2 / lambda_min;

dist_target = norm(config.s_target - r_center);
dist_interf = norm(config.s_interf - r_center);

fprintf('  Array diameter: %.3f m, max freq: %d Hz, lambda_min: %.3f m\n', ...
    config.array_diameter, f_max, lambda_min);
fprintf('  Fraunhofer distance: %.3f m\n', R_far);
fprintf('  Target distance: %.3f m %s\n', dist_target, ...
    iif(dist_target >= R_far, '✓ Far-field', '✗ Near-field'));
fprintf('  Interference distance: %.3f m %s\n', dist_interf, ...
    iif(dist_interf >= R_far, '✓ Far-field', '✗ Near-field'));

%% 4. Initialize signal convolution matrices
fprintf('[Load Data] Initializing signal matrices (before RIR convolution)\n');

X_target = zeros(Nt, Nmic);
X_interf = zeros(Nt, Nmic);
X_noisy = zeros(Nt, Nmic);

fprintf('[Load Data] Data loading complete.\n\n');

end

%% Helper function
function result = iif(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
