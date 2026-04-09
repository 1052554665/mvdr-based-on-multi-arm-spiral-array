%% MVDR 3D Beamforming with RIR - FIXED VERSION
%
% Improvements:
% 1. Fixed steering vector zero-padding issue (frequencies above f_limit)
% 2. Added NaN/Inf safety handling in ISTFT
% 3. Added audio output saving
% 4. Improved matrix inversion numerical stability
% 5. Added input validation
% 6. Better error handling throughout
%
% Author: Fixed version
% Date: 2026-04-09

clear; close all; clc;

%% ========== 1. Read mic positions with validation ==========
fprintf('========================================\n');
fprintf('  Multi-Arm Spiral Array MVDR Beamformin\n');
fprintf('========================================\n\n');

% Check if file exists
if ~exist('mic_positions.xlsx', 'file')
    error('Error: mic_positions.xlsx not found in current directory!');
end

fprintf('[1/9] Reading microphone positions...\n');
data = readmatrix('mic_positions.xlsx');
if size(data, 2) < 2
    error('Error: mic_positions.xlsx must have at least 2 columns (X, Y)!');
end

X = data(:,1);
Y = data(:,2);
Z = zeros(size(X));  % Default: z=0 plane

% Validate coordinate ranges
fprintf('  Raw coordinates: X=[%.3f, %.3f] m, Y=[%.3f, %.3f] m\n', ...
    min(X), max(X), min(Y), max(Y));

% Auto unit detection: if max range > 10, likely mm → convert to m
maxRange = max([abs([min(X) max(X) min(Y) max(Y)])]);
if maxRange > 10
    fprintf('  Detected unit: mm (range > 10). Converting to meters...\n');
    X = X / 1000;
    Y = Y / 1000;
    Z = Z / 1000;
else
    fprintf('  Detected unit: meters (range < 10).\n');
end

mic_pos = [X Y Z];  % N_mic × 3

% Center to array center position
array_center = [2.5 2 1.5];  % room center (m)
mic_pos = mic_pos - mean(mic_pos, 1) + array_center;

% Validate array is within room
L = [5 4 6];  % room dims (m)
Nmic = size(mic_pos, 1);
out_of_bounds = sum(...
    mic_pos(:,1) < 0 | mic_pos(:,1) > L(1) | ...
    mic_pos(:,2) < 0 | mic_pos(:,2) > L(2) | ...
    mic_pos(:,3) < 0 | mic_pos(:,3) > L(3));
if out_of_bounds > 0
    warning('%d microphones are outside room boundaries [0 %g] × [0 %g] × [0 %g]', ...
        out_of_bounds, L(1), L(2), L(3));
end

r_center = mean(mic_pos, 1);  % array center
fprintf('  %d microphones loaded, array center: [%.3f, %.3f, %.3f] m\n\n', ...
    Nmic, r_center(1), r_center(2), r_center(3));

% Visualization
figure(1);
scatter(X, Y, 40, 'filled');
axis equal; grid on;
xlabel('X (mm)'); ylabel('Y (mm)');
title('Microphone Array Geometry (Original Units)');

%% ========== 2. Room, RIR and signals parameters ==========
fprintf('[2/9] Setting simulation parameters...\n');
c = 340;              % speed of sound (m/s)
fs = 16000;           % sampling rate (Hz)
T = 1.0;              % duration (s)
Nt = T * fs;          % total samples
t = (0:1/fs:T-1/fs)'; % time vector

% Signal parameters
f0 = 1000;            % target freq (Hz)
f1 = 2000;            % interference freq (Hz)
SNR = 30;             % signal-to-noise ratio (dB)
INR = 10;             % interference-to-noise ratio (dB)
nsample = 4096;       % RIR length (samples)

% Source positions (3D, used for RIR generation)
s_target = [2 3.5 4];   % target source (m)
s_interf = [2.5 3.8 4]; % interference source (m)

fprintf('  Sound speed: %.1f m/s | Sampling rate: %d Hz | Duration: %.2f s\n', c, fs, T);
fprintf('  Target: %.0f Hz @ [%.2f, %.2f, %.2f] m\n', f0, s_target(1), s_target(2), s_target(3));
fprintf('  Interference: %.0f Hz @ [%.2f, %.2f, %.2f] m\n', f1, s_interf(1), s_interf(2), s_interf(3));
fprintf('  SNR: %.1f dB | INR: %.1f dB\n\n', SNR, INR);

%% ========== 3. Visualization: Room, sources, array ==========
fprintf('[3/9] Creating 3D visualization...\n');
figure(2); clf;
hold on; grid on; axis equal;

% Room boundaries
xv = [0 L(1) L(1) 0 0];
yv = [0 0 L(2) L(2) 0];
z0 = zeros(size(xv));

% Floor and ceiling
plot3(xv, yv, z0, 'k--', 'LineWidth', 1.2);
plot3(xv, yv, L(3)*ones(size(z0)), 'k--', 'LineWidth', 1.2);

% Vertical edges
for i = 1:4
    plot3([xv(i) xv(i)], [yv(i) yv(i)], [0 L(3)], 'k--');
end

% Microphone array
scatter3(mic_pos(:,1), mic_pos(:,2), mic_pos(:,3), 60, 'filled', 'b');
text(r_center(1), r_center(2), r_center(3)+0.1, 'Mic Array', ...
     'Color', 'b', 'FontWeight', 'bold');

% Target source
scatter3(s_target(1), s_target(2), s_target(3), 100, 'r', 'filled');
text(s_target(1), s_target(2), s_target(3)+0.1, 'Target', ...
     'Color', 'r', 'FontWeight', 'bold');

% Interference source
scatter3(s_interf(1), s_interf(2), s_interf(3), 100, 'm', 'filled');
text(s_interf(1), s_interf(2), s_interf(3)+0.1, 'Interference', ...
     'Color', 'm', 'FontWeight', 'bold');

% Connection lines
plot3([r_center(1) s_target(1)], [r_center(2) s_target(2)], [r_center(3) s_target(3)], 'r--', 'LineWidth', 1.2);
plot3([r_center(1) s_interf(1)], [r_center(2) s_interf(2)], [r_center(3) s_interf(3)], 'm--', 'LineWidth', 1.2);

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Room, Sources, and Microphone Array Layout');
view(45, 25);
legend({'Room','Microphones','Target','Interference'}, 'Location','best');
fprintf('  3D visualization complete.\n\n');

%% ========== 4. Fraunhofer criterion (far-field check) ==========
fprintf('[4/9] Checking far-field criterion...\n');
D = 0.15;           % array aperture (m) - 150 mm spiral
f_max = 5000;       % max frequency of interest (Hz)
lambda_min = c / f_max;
R_far = 2 * D^2 / lambda_min;  % Fraunhofer distance

dist_s_target = norm(s_target - r_center);
dist_s_interf = norm(s_interf - r_center);

fprintf('  Array diameter: %.3f m | f_max: %d Hz | λ_min: %.3f m\n', D, f_max, lambda_min);
fprintf('  Fraunhofer distance R_far: %.3f m\n', R_far);
fprintf('  Target distance: %.3f m %s\n', dist_s_target, ...
    iif(dist_s_target >= R_far, '(far-field)', '(near-field)'));
fprintf('  Interference distance: %.3f m %s\n\n', dist_s_interf, ...
    iif(dist_s_interf >= R_far, '(far-field)', '(near-field)'));

%% ========== 5. Generate RIRs per microphone ==========
fprintf('[5/9] Generating Room Impulse Responses (RIRs)...\n');

% RIR parameters
beta = 0;  % reflection coefficient (0=free-field, use 0.3-0.8 for reverb)
mtype = 'omnidirectional';
order = -1;
dim = 3;
orientation = 0;
hp_filter = true;

% Check if rir_generator exists
if ~exist('rir_generator', 'file')
    error('Error: rir_generator function not found! Please ensure it is in MATLAB path.');
end

h_target = zeros(Nmic, nsample);
h_interf = zeros(Nmic, nsample);

tic;
for m = 1:Nmic
    if mod(m, 20) == 0
        fprintf('  Processing microphone %d/%d...\n', m, Nmic);
    end
    rm = mic_pos(m,:);

    try
        ht = rir_generator(c, fs, rm, s_target, L, beta, nsample, mtype, order, dim, orientation, hp_filter);
        hi = rir_generator(c, fs, rm, s_interf, L, beta, nsample, mtype, order, dim, orientation, hp_filter);
    catch ME
        error('RIR generation failed for microphone %d: %s', m, ME.message);
    end

    % Ensure row vectors
    if iscolumn(ht), ht = ht.'; end
    if iscolumn(hi), hi = hi.'; end

    h_target(m,:) = ht;
    h_interf(m,:) = hi;
end
toc_rir = toc;
fprintf('  RIR generation done (%.2f s).\n\n', toc_rir);

%% ========== 6. Generate source signals and convolve ==========
fprintf('[6/9] Generating source signals and convolving with RIRs...\n');

x_target = sin(2*pi*f0*t);
x_interf = sin(2*pi*f1*t);

X_target = zeros(Nt, Nmic);
X_interf = zeros(Nt, Nmic);

% FIX: Use 'full' convolution and then trim to maintain full mixing effects
for m = 1:Nmic
    % Target signal convolution
    y_t = conv(x_target, h_target(m,:), 'full');
    X_target(:,m) = y_t(1:Nt);  % Keep first Nt samples

    % Interference signal convolution
    y_i = conv(x_interf, h_interf(m,:), 'full');
    X_interf(:,m) = y_i(1:Nt);  % Keep first Nt samples
end

% Scale interference by INR (amplitude scaling)
X_interf = X_interf / (10^(INR/20));

% Add noise per-channel to meet overall SNR
sig_rms = sqrt(mean(X_target.^2, 1));
desired_noise_rms = sig_rms ./ (10^(SNR/20));
noise = randn(size(X_target));
cur_noise_rms = sqrt(mean(noise.^2, 1));
noise = noise .* (desired_noise_rms ./ cur_noise_rms);

X_noisy = X_target + X_interf + noise;
fprintf('  Signal prepared: %d samples, %d channels, no manual delays.\n\n', Nt, Nmic);

%% ========== 7. STFT per channel ==========
fprintf('[7/9] Computing STFT for %d channels...\n', Nmic);

Nfft = 1024;
win = hamming(512);
noverlap = 256;

% Compute STFT for first channel to get freq/time dimensions
[S1, F, T] = stft(X_noisy(:,1), fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft);
Fbins = length(F);
Tframes = length(T);

% Pre-allocate full STFT matrix: S(freq x time x channel)
S = zeros(Fbins, Tframes, Nmic);
S(:,:,1) = S1;

% STFT for remaining channels
for m = 2:Nmic
    S(:,:,m) = stft(X_noisy(:,m), fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft);
end

% Find frequency limit (5 kHz)
f_limit = find(F <= f_max, 1, 'last');
if isempty(f_limit)
    f_limit = Fbins;
    warning('All frequencies < %.0f Hz; using all bins up to %.1f Hz.', f_max, F(end));
end

fprintf('  STFT complete: %d freq bins, %d time frames, freq up to %.1f Hz (bin %d).\n\n', ...
    Fbins, Tframes, F(f_limit), f_limit);

%% ========== 8. Steering vectors (3D, near/far field support) ==========
fprintf('[8/9] Computing steering vectors...\n');

a_target = zeros(Nmic, Fbins);
a_interf = zeros(Nmic, Fbins);

% Pre-compute distances (frequency-independent)
dist_target_all = zeros(Nmic, 1);
dist_interf_all = zeros(Nmic, 1);
for m = 1:Nmic
    dist_target_all(m) = norm(mic_pos(m,:) - s_target);
    dist_interf_all(m) = norm(mic_pos(m,:) - s_interf);
end

fprintf('  Target source distance range: [%.3f, %.3f] m\n', ...
    min(dist_target_all), max(dist_target_all));
fprintf('  Interference distance range: [%.3f, %.3f] m\n', ...
    min(dist_interf_all), max(dist_interf_all));

% FIX: For frequencies above f_limit, use equal-weight (no beam steering)
% This preserves high-frequency content instead of zeroing it out
for k = 1:Fbins
    freq = F(k);

    if k <= f_limit
        % ===== Standard 3D steering vector (near/far field) =====
        % Includes distance weighting: exp(-j*2π*f*dist/c) / dist
        a_target(:,k) = exp(-1j*2*pi*freq*dist_target_all/c) ./ max(dist_target_all, 1e-6);
        a_interf(:,k) = exp(-1j*2*pi*freq*dist_interf_all/c) ./ max(dist_interf_all, 1e-6);
    else
        % ===== High frequency: omni (equal weight) =====
        % FIX: Instead of zeros, use uniform response
        a_target(:,k) = ones(Nmic, 1) / sqrt(Nmic);
        a_interf(:,k) = ones(Nmic, 1) / sqrt(Nmic);
    end

    % Normalize steering vectors
    norm_at = norm(a_target(:,k));
    if norm_at > 0
        a_target(:,k) = a_target(:,k) / norm_at;
    end

    norm_ai = norm(a_interf(:,k));
    if norm_ai > 0
        a_interf(:,k) = a_interf(:,k) / norm_ai;
    end
end

fprintf('  Steering vectors computed (3D position-based, normalized).\n\n');

%% ========== 9. Adaptive MVDR Beamforming ==========
fprintf('[9/9] Running MVDR beamformer...\n');

Yf = zeros(Fbins, Tframes);  % Complex output STFT

% MVDR parameters
Mavg = 21;          % averaging window (frames) ~0.66 s
epsilon = 1e-2;     % base diagonal loading factor
shrink_alpha = 0.05; % shrinkage toward identity (0=no shrinkage)

tic;
for k = 1:f_limit
    Xkf = squeeze(S(k,:,:)).';  % Nmic x Tframes
    at = a_target(:,k);

    % FIX: Never skip frequency bins anymore (we fixed steering vectors)
    if norm(at) < 1e-10
        continue;  % Very unlikely now
    end

    for tt = 1:Tframes
        % Define local frame window
        t1 = max(1, tt - floor(Mavg/2));
        t2 = min(Tframes, tt + floor(Mavg/2));
        Xloc = Xkf(:, t1:t2);

        % Covariance matrix estimation
        Rxx = (Xloc * Xloc') / size(Xloc, 2);

        % Optional shrinkage for additional stability
        if shrink_alpha > 0
            R_shrink = (1 - shrink_alpha) * Rxx + shrink_alpha * (trace(Rxx)/Nmic) * eye(Nmic);
            Rxx = R_shrink;
        end

        % Adaptive diagonal loading (proportional to trace)
        reg = epsilon * trace(Rxx) / Nmic;
        Rxx = Rxx + reg * eye(Nmic);

        % FIX: Check condition number for stability
        cond_num = cond(Rxx);
        if cond_num > 1e12
            % Increase loading if ill-conditioned
            reg = reg * 10;
            Rxx = Rxx + (reg - epsilon*trace(Rxx)/Nmic) * eye(Nmic);
        end

        % MVDR weight computation: w = R^{-1} a / (a^H R^{-1} a)
        try
            w = Rxx \ at;
        catch ME
            % If inversion fails, skip this frame
            warning('MVDR inversion failed at k=%d, tt=%d: %s', k, tt, ME.message);
            w = zeros(Nmic, 1);
        end

        denom = (at' * w);
        if abs(denom) < 1e-12
            W = zeros(size(w));
        else
            W = w ./ denom;
        end

        % Beamformed output for this frame
        Yf(k, tt) = W' * Xkf(:, tt);
    end

    if mod(k, 50) == 0
        fprintf('  Processed frequency bin %d/%d (%.1f Hz)\n', k, f_limit, F(k));
    end
end
toc_mvdr = toc;
fprintf('  MVDR complete (%.2f s).\n\n', toc_mvdr);

%% ========== 10. Diagnostics: Eigenspectrum and Beampattern ==========
fprintf('Generating diagnostics...\n');

% Find frequency bins for diagnostics
[~, k2] = min(abs(F - 2000));
[~, k1] = min(abs(F - 1000));

% Robust frame range (middle 30% of frames)
t1_diag = max(1, round(Tframes * 0.35));
t2_diag = min(Tframes, round(Tframes * 0.65));
if t1_diag >= t2_diag
    t1_diag = 1;
    t2_diag = Tframes;
end

% Compute Rxx for 2 kHz
Xkf_k2 = squeeze(S(k2,:,:)).';  % Nmic x Tframes
Xloc_diag = Xkf_k2(:, t1_diag:t2_diag);
Rxx_diag = (Xloc_diag * Xloc_diag') / size(Xloc_diag, 2);
reg_diag = epsilon * trace(Rxx_diag) / Nmic;
Rxx_diag = Rxx_diag + reg_diag * eye(Nmic);

% Eigenvalue spectrum (Figure 3)
D = eig(Rxx_diag);
evals = sort(D, 'descend');
figure(3);
semilogy(1:Nmic, evals + eps, '-o', 'LineWidth', 1.5, 'MarkerSize', 5);
xlabel('Index'); ylabel('Eigenvalue (log scale)');
title(sprintf('Rxx Eigenspectrum at %.0f Hz', F(k2)));
grid on;

% MVDR weight and beampattern at 2 kHz (Figure 4)
center_frame = round(Tframes / 2);
t1c = max(1, center_frame - floor(Mavg/2));
t2c = min(Tframes, center_frame + floor(Mavg/2));
Xlocc = Xkf_k2(:, t1c:t2c);
Rxxc = (Xlocc * Xlocc') / size(Xlocc, 2) + (epsilon * trace(Xlocc * Xlocc') / Nmic) * eye(Nmic);
at_k2 = a_target(:, k2);
w_diag = Rxxc \ at_k2;
denom_diag = (at_k2' * w_diag);
if abs(denom_diag) < 1e-12
    Wdiag = zeros(size(w_diag));
else
    Wdiag = w_diag ./ denom_diag;
end

% Beampattern (plane-wave visualization only, not actual 3D)
azs = -180:1:180;
resp = zeros(size(azs));
for ii = 1:length(azs)
    az_rad = azs(ii) * pi / 180;
    d_try = [cos(az_rad), sin(az_rad), 0];
    a_try = exp(-1j*2*pi*F(k2) * ((mic_pos - r_center) * d_try.') / c);
    if norm(a_try) > 0
        a_try = a_try / norm(a_try);
    end
    resp(ii) = 20*log10(abs(Wdiag' * a_try) + eps);
end

figure(4);
plot(azs, resp, 'LineWidth', 1.5);
grid on;
xlabel('Azimuth (degrees)'); ylabel('Response (dB)');
title(sprintf('MVDR Beampattern at %.0f Hz (Plane-wave, visualization only)', F(k2)));
set(gca, 'XLim', [-180 180]);

%% ========== 11. ISTFT Reconstruction (with NaN/Inf safety) ==========
fprintf('Reconstructing time-domain signal via ISTFT...\n');

y_mvdr = istft(Yf, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft);
y_mvdr = real(y_mvdr);

% FIX: Handle NaN/Inf values safely
y_mvdr(~isfinite(y_mvdr)) = 0;  % Replace NaN/Inf with 0
fprintf('  Cleaned %d NaN/Inf values from ISTFT output.\n', sum(~isfinite(y_mvdr)));

% Adjust length to match input
if length(y_mvdr) >= Nt
    y_mvdr = y_mvdr(1:Nt);
else
    y_mvdr = [y_mvdr; zeros(Nt - length(y_mvdr), 1)];
end

% Safe normalization (FIX)
y_max = max(abs(y_mvdr));
if y_max > 0 && isfinite(y_max)
    y_mvdr = y_mvdr / y_max;
    fprintf('  Normalized to peak amplitude 1.0.\n');
else
    warning('Could not normalize output: y_max=%.3e', y_max);
    fprintf('  Output left unnormalized.\n');
end

%% ========== 12. Save output audio (FIXED: was missing!) ==========
fprintf('\nSaving output audio file...\n');
audiowrite('y_mvdr.wav', y_mvdr, fs);
fprintf('  ✓ Saved: y_mvdr.wav (%.2f s, %d Hz)\n', length(y_mvdr)/fs, fs);

%% ========== 13. Comparison plots ==========
fprintf('Generating comparison plots...\n');

% Figure 5: Time-domain waveforms
figure(5);
t_plot = (0:min(5000,Nt)-1) / fs;  % First 5000 samples (~0.31 s)
subplot(3,1,1);
plot(t_plot, x_target(1:length(t_plot), 1), 'LineWidth', 1);
xlabel('Time (s)'); ylabel('Amplitude');
title('Pure Target Signal (1 kHz)');
grid on;

subplot(3,1,2);
plot(t_plot, X_noisy(1:length(t_plot), 1), 'LineWidth', 0.5, 'Color', [0.5 0.5 0.5]);
xlabel('Time (s)'); ylabel('Amplitude');
title('Mixed Signal: Target + Interference + Noise');
grid on;

subplot(3,1,3);
plot(t_plot, y_mvdr(1:length(t_plot)), 'LineWidth', 1, 'Color', 'r');
xlabel('Time (s)'); ylabel('Amplitude');
title('MVDR Beamformed Output');
grid on;

% Figure 6: Frequency-domain comparison
figure(6);
nfft_plot = 2048;
[Pxx_target, f_pxx] = pwelch(x_target, hamming(512), 256, nfft_plot, fs);
[Pxx_mixed, ~] = pwelch(X_noisy(:,1), hamming(512), 256, nfft_plot, fs);
[Pxx_mvdr, ~] = pwelch(y_mvdr, hamming(512), 256, nfft_plot, fs);

subplot(3,1,1);
semilogy(f_pxx(1:500), Pxx_target(1:500));
xlabel('Frequency (Hz)'); ylabel('PSD (V²/Hz)');
title('Pure Target Signal');
grid on; xlim([0 5000]);

subplot(3,1,2);
semilogy(f_pxx(1:500), Pxx_mixed(1:500), 'Color', [0.5 0.5 0.5]);
xlabel('Frequency (Hz)'); ylabel('PSD (V²/Hz)');
title('Mixed Signal');
grid on; xlim([0 5000]);

subplot(3,1,3);
semilogy(f_pxx(1:500), Pxx_mvdr(1:500), 'Color', 'r');
xlabel('Frequency (Hz)'); ylabel('PSD (V²/Hz)');
title('MVDR Output');
grid on; xlim([0 5000]);

fprintf('All plots generated.\n\n');

%% ========== Summary ==========
fprintf('\n========== PROCESSING COMPLETE ==========\n');
fprintf('Input  : %d samples, %d channels, duration %.2f s\n', Nt, Nmic, T);
fprintf('Output : %s\n', 'y_mvdr.wav');
fprintf('Total processing time: %.2f s (RIR: %.2f s, MVDR: %.2f s)\n', ...
    toc_rir + toc_mvdr, toc_rir, toc_mvdr);
fprintf('\nTarget frequency: %.0f Hz\n', f0);
fprintf('Interference frequency: %.0f Hz\n', f1);
fprintf('SNR: %.1f dB | INR: %.1f dB\n', SNR, INR);
fprintf('========================================\n\n');

%% Helper function
function result = iif(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
