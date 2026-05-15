%% ============== Diagnostics and Visualization Module ==============
% Computes diagnostics (SNR improvement, beampatterns) and generates visualizations

function diagnostics_and_visualization_module(config, Yf_tgt, Y_tar_tgt, Y_intnoi_tgt, ...
    Yf_int, Y_interf_int, Y_tarnoi_int, S, S_interf, S_intnoi, S_tarnoi, ...
    X_noisy, X_target, X_interf, F, T, a_target, a_interf, mic_pos, r_center, Nt, Nmic)

fprintf('\n[Diagnostics] Starting diagnostics and visualization...\n');
% Improve visibility: bold black axes/text by default for generated figures
set(groot, 'defaultAxesFontWeight', 'bold', 'defaultAxesFontSize', 12, ...
    'defaultTextColor', 'k', 'defaultAxesXColor', 'k', 'defaultAxesYColor', 'k', 'defaultAxesZColor', 'k');

%% Define figure output paths
% figures_dir = fullfile(config.output_dir, 'demo');
figures_dir = fullfile(config.output_dir, 'figures');
path_eigenspectrum_2kHz = fullfile(figures_dir, 'Rxx_eigenspectrum_2kHz.pdf');
path_beampattern_2kHz = fullfile(figures_dir, 'Beampattern_2kHz.pdf');
path_target_signal = fullfile(figures_dir, 'Target_signal.pdf');
path_received_signal = fullfile(figures_dir, 'Received_signal.pdf');
path_mvdr_outputs = fullfile(figures_dir, 'MVDR_outputs.pdf');
path_signal_comparison = fullfile(figures_dir, 'Signal_comparison_input_vs_mvdr.pdf');
path_spectrograms = fullfile(figures_dir, 'Signal_comparison_spectrograms_input_vs_mvdr.pdf');
path_performance_improvement = fullfile(figures_dir, 'Performance_improvement.pdf');
path_psd_fullband = fullfile(figures_dir, 'PSD_fullband.pdf');
path_psd_4kHz = fullfile(figures_dir, 'PSD_4kHz.pdf');
path_mic_array_2d = fullfile(figures_dir, 'Microphone_array_geometry_2d.pdf');
path_room_layout_3d = fullfile(figures_dir, 'Room_layout_with_sources_3d.pdf');

%% Unified font size settings for publication-quality figures (IEEE standard)
font_sz.title = 32;          % Main titles for single plots
font_sz.subtitle = 31;       % Subtitle/axis labels for single plots
font_sz.label = 31;          % Axis labels (xlabel, ylabel)
font_sz.subtile_title = 31;  % Titles in subplots (tiledlayout)
font_sz.legend = 30;         % Legend text
font_sz.tick = 30;           % Tick labels
font_sz.small_tick = 29;      % Small tick labels for dense subplots
font_sz.colorbar = 30;       % Colorbar tick labels
font_sz.colorbar_label = 30; % Colorbar label
%% Figure save helper function
save_figure = @(fig_path) save_fig_func(config, fig_path);
    function save_fig_func(config, fig_path)
        if config.save_figures
            set(gcf,'Color','white');
            set(gca,'Color','white');
            exportgraphics(gcf, fig_path, ...
                'ContentType','vector', 'BackgroundColor','white', 'Resolution',600);
        end
    end

win = hamming(config.win_len);
ref_mic = config.ref_mic;

%% 0. Array and Room Visualization
fprintf('[Diagnostics] Generating array and room visualizations...\n');

% 2D Microphone Array Scatter Plot
figure(200); clf;
scatter(mic_pos(:,1), mic_pos(:,2), 40, 'filled', 'k');
axis equal;
xlabel('X (m)', 'FontSize', font_sz.label);
ylabel('Y (m)', 'FontSize', font_sz.label);
% title('Microphone Array Geometry (Top View)', 'FontSize', font_sz.title);
grid on;
set(gca, 'LineWidth', 1, 'FontSize', font_sz.tick);
set(gcf, 'Color', 'white');
set(gca, 'Color', 'white');
save_figure(path_mic_array_2d);

% 3D Room, Array, and Source Positions Visualization
figure(201); clf;
hold on; grid on; axis equal;

% Room boundaries
room_L = config.room_L;  % [length, width, height]
xv = [0 room_L(1) room_L(1) 0 0];
yv = [0 0 room_L(2) room_L(2) 0];
z0 = zeros(size(xv));

% Floor and ceiling
plot3(xv, yv, z0, 'k--', 'LineWidth', 1.2);
plot3(xv, yv, room_L(3)*ones(size(z0)), 'k--', 'LineWidth', 1.2);

% Vertical edges
for i = 1:4
    plot3([xv(i) xv(i)], [yv(i) yv(i)], [0 room_L(3)], 'k--', 'LineWidth', 0.8);
end

% Microphone array
scatter3(mic_pos(:,1), mic_pos(:,2), mic_pos(:,3), 60, 'filled', 'b');
text(r_center(1), r_center(2), r_center(3)+0.2, 'Mic Array', ...
     'Color', 'b', 'FontWeight', 'bold', 'FontSize', font_sz.label);

% Target and interference sources
s_target = config.s_target;
s_interf = config.s_interf;
scatter3(s_target(1), s_target(2), s_target(3), 100, 'r', 'filled');
text(s_target(1)+0.1, s_target(2), s_target(3)+0.15, 'Target', ...
    'Color', 'r', 'FontWeight', 'bold', 'FontSize', font_sz.label);

scatter3(s_interf(1), s_interf(2), s_interf(3), 100, 'm', 'filled');
text(s_interf(1)+0.1, s_interf(2), s_interf(3)+0.15, 'Interference', ...
    'Color', 'm', 'FontWeight', 'bold', 'FontSize', font_sz.label);

% Connection lines from array center to sources
plot3([r_center(1) s_target(1)], [r_center(2) s_target(2)], [r_center(3) s_target(3)], 'r--', 'LineWidth', 1.2);
plot3([r_center(1) s_interf(1)], [r_center(2) s_interf(2)], [r_center(3) s_interf(3)], 'm--', 'LineWidth', 1.2);

xlabel('X (m)', 'FontSize', font_sz.label);
ylabel('Y (m)', 'FontSize', font_sz.label);
zlabel('Z (m)', 'FontSize', font_sz.label);
view(45, 25);
set(gca, 'LineWidth', 1, 'FontSize', font_sz.tick);
set(gcf, 'Color', 'white');
set(gca, 'Color', 'white');
save_figure(path_room_layout_3d);

%% 1. Inverse STFT
fprintf('[Diagnostics] Computing inverse STFT...\n');

y_mvdr_target = real(istft(Yf_tgt, config.fs, ...
    'Window', win, 'OverlapLength', config.noverlap, 'FFTLength', config.Nfft));
y_mvdr_interf = real(istft(Yf_int, config.fs, ...
    'Window', win, 'OverlapLength', config.noverlap, 'FFTLength', config.Nfft));

y_tar_out_target = real(istft(Y_tar_tgt, config.fs, ...
    'Window', win, 'OverlapLength', config.noverlap, 'FFTLength', config.Nfft));
y_intnoi_out_target = real(istft(Y_intnoi_tgt, config.fs, ...
    'Window', win, 'OverlapLength', config.noverlap, 'FFTLength', config.Nfft));

y_interf_out_interf = real(istft(Y_interf_int, config.fs, ...
    'Window', win, 'OverlapLength', config.noverlap, 'FFTLength', config.Nfft));
y_tarnoi_out_interf = real(istft(Y_tarnoi_int, config.fs, ...
    'Window', win, 'OverlapLength', config.noverlap, 'FFTLength', config.Nfft));

% Length alignment
Lsig = min([length(y_mvdr_target), length(y_mvdr_interf), ...
    length(y_tar_out_target), length(y_intnoi_out_target), ...
    length(y_interf_out_interf), length(y_tarnoi_out_interf)]);

y_mvdr_target = y_mvdr_target(1:Lsig);
y_mvdr_interf = y_mvdr_interf(1:Lsig);
y_tar_out_target = y_tar_out_target(1:Lsig);
y_intnoi_out_target = y_intnoi_out_target(1:Lsig);
y_interf_out_interf = y_interf_out_interf(1:Lsig);
y_tarnoi_out_interf = y_tarnoi_out_interf(1:Lsig);

% Normalize
y_mvdr_target = y_mvdr_target / max(abs(y_mvdr_target) + 1e-12);
y_mvdr_interf = y_mvdr_interf / max(abs(y_mvdr_interf) + 1e-12);

%% 2. Performance metrics - Target-steered branch
fprintf('[Diagnostics] Computing performance metrics...\n');

P_tar_out = mean(y_tar_out_target.^2);
P_intnoi_out = mean(y_intnoi_out_target.^2);
SNR_out_dB = 10*log10(P_tar_out / (P_intnoi_out + eps));

x_tar_in = X_target(:, ref_mic);
P_tar_in = mean(x_tar_in.^2);
x_int_in = X_interf(:, ref_mic);
P_int_in = mean(x_int_in.^2);
SNR_in_dB = 10*log10(P_tar_in / (P_int_in + eps));

SNR_gain_dB = SNR_out_dB - SNR_in_dB;

fprintf('\n===== TARGET-STEERED MVDR =====\n');
fprintf('  Input  SINR:  %.2f dB\n', SNR_in_dB);
fprintf('  Output SINR:  %.2f dB\n', SNR_out_dB);
fprintf('  SNR Gain:     %.2f dB\n', SNR_gain_dB);
fprintf('================================\n');

%% 3. Performance metrics - Interference-steered branch
P_interf_in = mean(X_interf(:, ref_mic).^2);
P_tarnoi_in = mean(X_target(:, ref_mic).^2);
ISR_in_dB = 10*log10(P_interf_in / (P_tarnoi_in + eps));

P_interf_out = mean(y_interf_out_interf.^2);
P_tarnoi_out = mean(y_tarnoi_out_interf.^2);
ISR_out_dB = 10*log10(P_interf_out / (P_tarnoi_out + eps));
ISR_gain_dB = ISR_out_dB - ISR_in_dB;

fprintf('\n===== INTERFERENCE-STEERED MVDR =====\n');
fprintf('  Input  ISR:   %.2f dB\n', ISR_in_dB);
fprintf('  Output ISR:   %.2f dB\n', ISR_out_dB);
fprintf('  ISR Gain:     %.2f dB\n', ISR_gain_dB);
fprintf('========================================\n');

%% 4. Beampattern diagnostics at key frequencies
fprintf('[Diagnostics] Generating beampattern diagnostics...\n');

[~, k2k] = min(abs(F - 2000));
[~, k4k] = min(abs(F - 4000));

% 2 kHz beampattern and eigenspectrum
numFrames = size(Yf_tgt, 2);
t1_diag = max(1, round(numFrames*0.35));
t2_diag = min(numFrames, round(numFrames*0.65));

Xkf_k2 = squeeze(S(k2k,:,:)).';
t2_diag = min(t2_diag, size(Xkf_k2,2));
Xloc_diag = Xkf_k2(:, t1_diag:t2_diag);
Rxx_diag = (Xloc_diag * Xloc_diag') / size(Xloc_diag,2);
reg_diag = config.epsilon * trace(Rxx_diag) / Nmic;
Rxx_diag = Rxx_diag + reg_diag * eye(Nmic);

[~, D] = eig(Rxx_diag);
evals = sort(diag(D), 'descend');

figure(203); clf;
plot(1:Nmic, 10*log10(evals + eps), '-o', 'LineWidth', 2, 'MarkerSize', 8, 'Color', 'k');
xlabel('Index', 'FontSize', font_sz.label); ylabel('Eigenvalue (dB)', 'FontSize', font_sz.label);
% title(sprintf('Covariance Eigenspectrum at %.0f Hz', F(k2k)));
grid on;
set(gca, 'LineWidth', 1.5, 'FontSize', font_sz.tick);
save_figure(path_eigenspectrum_2kHz);

% Beampattern at 2 kHz
center_frame = round(numFrames/2);
t1c = max(1, center_frame - floor(config.Mavg/2));
t2c = min(numFrames, center_frame + floor(config.Mavg/2));
Xlocc = Xkf_k2(:, t1c:t2c);
Rxxc = (Xlocc * Xlocc') / size(Xlocc,2);

if config.shrink_alpha > 0
    mu_c = trace(Rxxc) / Nmic;
    Rxxc = (1 - config.shrink_alpha) * Rxxc + config.shrink_alpha * mu_c * eye(Nmic);
end
Rxxc = Rxxc + (config.epsilon * trace(Rxxc) / Nmic) * eye(Nmic);

at_k2 = a_target(:, k2k);
w_diag = Rxxc \ at_k2;
denom_diag = at_k2' * w_diag;
if abs(denom_diag) < 1e-12
    Wdiag = zeros(size(w_diag));
else
    Wdiag = w_diag / denom_diag;
end

azs = -180:1:180;
resp_2k = zeros(size(azs));
for ii = 1:length(azs)
    d_try = [cosd(0)*cosd(azs(ii)); cosd(0)*sind(azs(ii)); sind(0)];
    a_try = exp(-1j*2*pi*F(k2k) * ((mic_pos - r_center) * d_try) / config.c);
    a_try = a_try / norm(a_try);
    resp_2k(ii) = 20*log10(abs(Wdiag' * a_try) + eps);
end

figure(204); clf;
plot(azs, resp_2k, 'LineWidth', 2, 'Color', 'k');
xlabel('Azimuth (deg)', 'FontSize', font_sz.label); ylabel('Response (dB)', 'FontSize', font_sz.label);
% title(sprintf('Target-Steered Beampattern at %.0f Hz', F(k2k)));
grid on;
set(gca, 'LineWidth', 1.5, 'FontSize', font_sz.tick);
save_figure(path_beampattern_2kHz);

%% 5. Beampattern and diagnostics at 4 kHz (interference)
fprintf('[Diagnostics] Computing 4 kHz diagnostics (interference frequency)...\n');

if config.use_oracle_intnoi_cov
    Xkf_k4 = squeeze(S_intnoi(k4k,:,:)).';
else
    Xkf_k4 = squeeze(S(k4k,:,:)).';
end

t1c_4k = max(1, center_frame - floor(config.Mavg/2));
t2c_4k = min(numFrames, center_frame + floor(config.Mavg/2));
t2c_4k = min(t2c_4k, size(Xkf_k4,2));
Xlocc_4k = Xkf_k4(:, t1c_4k:t2c_4k);
Rxxc_4k = (Xlocc_4k * Xlocc_4k') / size(Xlocc_4k,2);

if config.shrink_alpha > 0
    mu_c_4k = trace(Rxxc_4k) / Nmic;
    Rxxc_4k = (1 - config.shrink_alpha) * Rxxc_4k + config.shrink_alpha * mu_c_4k * eye(Nmic);
end

epsilon_4k = config.epsilon * 0.1;
Rxxc_4k = Rxxc_4k + (epsilon_4k * trace(Rxxc_4k) / Nmic) * eye(Nmic);

at_k4 = a_target(:, k4k);
w_diag_4k = Rxxc_4k \ at_k4;
denom_diag_4k = at_k4' * w_diag_4k;
if abs(denom_diag_4k) < 1e-12
    Wdiag_4k = zeros(size(w_diag_4k));
else
    Wdiag_4k = w_diag_4k / denom_diag_4k;
end

fprintf('\n===== 4 kHz Diagnostics =====\n');
fprintf('  Condition number: %.2e\n', cond(Rxxc_4k));
resp_target_4k = 20*log10(abs(Wdiag_4k' * at_k4) + eps);
ai_k4 = a_interf(:, k4k);
resp_interf_4k = 20*log10(abs(Wdiag_4k' * ai_k4) + eps);
fprintf('  Response at target DOA:   %.2f dB\n', resp_target_4k);
fprintf('  Response at interferer:   %.2f dB\n', resp_interf_4k);
fprintf('  Suppression:              %.2f dB\n', resp_target_4k - resp_interf_4k);
fprintf('==============================\n');

%% 6. Time-domain waveforms
fprintf('[Diagnostics] Generating waveform plots...\n');

figure(205); clf;
t_axis = (0:length(X_target(:,1))-1) / config.fs;
plot(t_axis, X_target(:,1), 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', font_sz.label); ylabel('Amplitude', 'FontSize', font_sz.label);
title('Target Signal (Mic 1)', 'FontSize', font_sz.title);
grid on;
set(gca, 'LineWidth', 1, 'FontSize', font_sz.tick);
save_figure(path_target_signal);

figure(206); clf;
plot(t_axis, X_noisy(:,1), 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', font_sz.label); ylabel('Amplitude', 'FontSize', font_sz.label);
title('Received Signal with Interference (Mic 1)', 'FontSize', font_sz.title);
grid on;
set(gca, 'LineWidth', 1, 'FontSize', font_sz.tick);
save_figure(path_received_signal);

figure(207); clf;
t_mvdr = (0:length(y_mvdr_target)-1) / config.fs;
plot(t_mvdr, y_mvdr_target, 'b', 'LineWidth', 1.5); hold on;
plot(t_mvdr, y_mvdr_interf, 'm', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', font_sz.label); ylabel('Amplitude', 'FontSize', font_sz.label);
% h = legend('2kHz-steered', '4kHz-steered', 'Location', 'best');
h = legend('Target-steered', 'Interference-steered', 'Location', 'best');
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', 'white', 'EdgeColor', 'k', 'FontSize', font_sz.legend);
% title('MVDR Beamformer Outputs', 'FontSize', font_sz.title);
grid on;
set(gca, 'LineWidth', 1, 'FontSize', font_sz.tick);
save_figure(path_mvdr_outputs);

% IEEE-style overview: one figure with input vs MVDR-processed waveforms
Lcmp = min([length(X_target(:,1)), length(X_interf(:,1)), length(X_noisy(:,1)), ...
            length(y_tar_out_target), length(y_interf_out_interf), length(y_mvdr_target), length(y_mvdr_interf)]);
t_cmp = (0:Lcmp-1) / config.fs;

x_target_cmp = X_target(1:Lcmp,1);
x_interf_cmp = X_interf(1:Lcmp,1);
x_noisy_cmp = X_noisy(1:Lcmp,1);

y_target_cmp = y_tar_out_target(1:Lcmp);
y_interf_cmp = y_interf_out_interf(1:Lcmp);
y_recv_cmp = y_mvdr_target(1:Lcmp) + y_mvdr_interf(1:Lcmp);

figure(211); clf;
tiledlayout(2,3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(t_cmp, x_target_cmp, 'b', 'LineWidth', 1.1);
t = title('Input Target (Mic 1)'); set(t, 'Color', 'k', 'FontName', 'Times New Roman'); xlabel('Time (s)'); ylabel('Amplitude'); grid on;

nexttile;
plot(t_cmp, x_interf_cmp, 'm', 'LineWidth', 1.1);
t = title('Input Interference (Mic 1)'); set(t, 'Color', 'k', 'FontName', 'Times New Roman'); xlabel('Time (s)'); ylabel('Amplitude'); grid on;

nexttile;
plot(t_cmp, x_noisy_cmp, 'k', 'LineWidth', 1.1);
t = title('Input Received (Mic 1)'); 
set(t, 'Color', 'k', 'FontName', 'Times New Roman'); 
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

nexttile;
plot(t_cmp, y_target_cmp, 'b', 'LineWidth', 1.1);
t = title('MVDR Processed Target'); 
set(t, 'Color', 'k', 'FontName', 'Times New Roman'); 
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

nexttile;
plot(t_cmp, y_interf_cmp, 'm', 'LineWidth', 1.1);
t = title('MVDR Processed Interference'); set(t, 'Color', 'k', 'FontName', 'Times New Roman'); xlabel('Time (s)'); ylabel('Amplitude'); grid on;

nexttile;
plot(t_cmp, y_recv_cmp, 'k', 'LineWidth', 1.1);
t = title('MVDR Processed Received'); set(t, 'Color', 'k', 'FontName', 'Times New Roman'); xlabel('Time (s)'); ylabel('Amplitude'); grid on;

% sg = sgtitle('Input vs MVDR-Processed Signals');
% set(sg, 'Color', 'k', 'FontWeight', 'bold', 'FontName', 'Times New Roman', 'FontSize', 20);
set(findall(gcf,'type','axes'), 'LineWidth', 1, 'FontSize', font_sz.small_tick);
set(gcf, 'Color', 'white');
set(findall(gcf,'type','axes'), 'Color', 'white', 'XColor', 'k', 'YColor', 'k');
save_figure(path_signal_comparison);

% IEEE-style overview: one figure with input vs MVDR-processed spectrograms
spec_win = hamming(256);
spec_overlap = 128;
spec_nfft = 512;

[S_xt, Fspec, Tspec] = spectrogram(X_target(1:Lcmp,1), spec_win, spec_overlap, spec_nfft, config.fs);
[S_xi, ~, ~] = spectrogram(X_interf(1:Lcmp,1), spec_win, spec_overlap, spec_nfft, config.fs);
[S_xr, ~, ~] = spectrogram(X_noisy(1:Lcmp,1), spec_win, spec_overlap, spec_nfft, config.fs);

[S_yt, ~, ~] = spectrogram(y_target_cmp, spec_win, spec_overlap, spec_nfft, config.fs);
[S_yi, ~, ~] = spectrogram(y_interf_cmp, spec_win, spec_overlap, spec_nfft, config.fs);
[S_yr, ~, ~] = spectrogram(y_recv_cmp, spec_win, spec_overlap, spec_nfft, config.fs);

P_xt = 10*log10(abs(S_xt).^2 + eps);
P_xi = 10*log10(abs(S_xi).^2 + eps);
P_xr = 10*log10(abs(S_xr).^2 + eps);
P_yt = 10*log10(abs(S_yt).^2 + eps);
P_yi = 10*log10(abs(S_yi).^2 + eps);
P_yr = 10*log10(abs(S_yr).^2 + eps);

all_spec_vals = [P_xt(:); P_xi(:); P_xr(:); P_yt(:); P_yi(:); P_yr(:)];
spec_clim_max = max(all_spec_vals);
spec_clim_min = spec_clim_max - 80;

figure(212); clf;
tiledlayout(2,3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
imagesc(Tspec, Fspec, P_xt); axis xy;
t = title('Input Target (Mic 1)'); set(t, 'Color', 'k', 'FontSize', font_sz.subtile_title); xlabel('Time (s)', 'FontSize', font_sz.subtitle); ylabel('Frequency (Hz)', 'FontSize', font_sz.subtitle);
ylim([0 config.fs/2]); caxis([spec_clim_min spec_clim_max]); set(gca, 'FontSize', font_sz.small_tick);

nexttile;
imagesc(Tspec, Fspec, P_xi); axis xy;
t = title('Input Interference (Mic 1)'); set(t, 'Color', 'k', 'FontSize', font_sz.subtile_title); xlabel('Time (s)', 'FontSize', font_sz.subtitle); ylabel('Frequency (Hz)', 'FontSize', font_sz.subtitle);
ylim([0 config.fs/2]); caxis([spec_clim_min spec_clim_max]); set(gca, 'FontSize', font_sz.small_tick);

nexttile;
imagesc(Tspec, Fspec, P_xr); axis xy;
t = title('Input Received (Mic 1)'); set(t, 'Color', 'k', 'FontSize', font_sz.subtile_title); xlabel('Time (s)', 'FontSize', font_sz.subtitle); ylabel('Frequency (Hz)', 'FontSize', font_sz.subtitle);
ylim([0 config.fs/2]); caxis([spec_clim_min spec_clim_max]); set(gca, 'FontSize', font_sz.small_tick);

nexttile;
imagesc(Tspec, Fspec, P_yt); axis xy;
t = title('MVDR Processed Target'); set(t, 'Color', 'k', 'FontSize', font_sz.subtile_title); xlabel('Time (s)', 'FontSize', font_sz.subtitle); ylabel('Frequency (Hz)', 'FontSize', font_sz.subtitle);
ylim([0 config.fs/2]); caxis([spec_clim_min spec_clim_max]); set(gca, 'FontSize', font_sz.small_tick);

nexttile;
imagesc(Tspec, Fspec, P_yi); axis xy;
t = title('MVDR Processed Interference'); set(t, 'Color', 'k', 'FontSize', font_sz.subtile_title); xlabel('Time (s)', 'FontSize', font_sz.subtitle); ylabel('Frequency (Hz)', 'FontSize', font_sz.subtitle);
ylim([0 config.fs/2]); caxis([spec_clim_min spec_clim_max]); set(gca, 'FontSize', font_sz.small_tick);

nexttile;
imagesc(Tspec, Fspec, P_yr); axis xy;
t = title('MVDR Processed Received'); set(t, 'Color', 'k', 'FontSize', font_sz.subtile_title); xlabel('Time (s)', 'FontSize', font_sz.subtitle); ylabel('Frequency (Hz)', 'FontSize', font_sz.subtitle);
ylim([0 config.fs/2]); caxis([spec_clim_min spec_clim_max]); set(gca, 'FontSize', font_sz.small_tick);

colormap(turbo);
set(findall(gcf,'type','axes'), 'LineWidth', 1, 'Color', 'white', 'XColor', 'k', 'YColor', 'k', 'FontSize', font_sz.small_tick);
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Power/Frequency (dB)';
cb.Color = 'k';
cb.FontWeight = 'bold';
cb.Label.Color = 'k';
cb.Label.FontWeight = 'bold';
cb.FontSize = font_sz.colorbar;
cb.Label.FontSize = font_sz.colorbar_label;
% sg = sgtitle('Input vs MVDR-Processed Spectrograms', 'Color', 'k', 'FontWeight', 'bold');
% set(sg, 'Color', 'k', 'FontName', 'Times New Roman', 'FontSize', 20);
set(gcf, 'Color', 'white');
save_figure(path_spectrograms);

%% 7. Performance bar chart
fprintf('[Diagnostics] Generating performance chart...\n');

figure(208); clf;
bar([SNR_in_dB, SNR_out_dB; ISR_in_dB, ISR_out_dB]);
set(gca, 'XTickLabel', {'2kHz-steered', '4kHz-steered'}, 'FontSize', font_sz.tick);
ylabel('Ratio (dB)', 'FontSize', font_sz.label);
h = legend('Before MVDR', 'After MVDR', 'Location', 'best');
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', 'white', 'EdgeColor', 'k', 'FontSize', font_sz.legend);
grid on;
set(gca, 'LineWidth', 1, 'FontSize', font_sz.tick);
save_figure(path_performance_improvement);

%% 8. PSD comparison
fprintf('[Diagnostics] Computing power spectral densities...\n');

nfft_psd = 4096;
[Pxx_in, Fp] = pwelch(X_noisy(:,1), hann(1024), 512, nfft_psd, config.fs);
[Pxx_out_target, ~] = pwelch(y_mvdr_target, hann(1024), 512, nfft_psd, config.fs);
[Pxx_out_interf, ~] = pwelch(y_mvdr_interf, hann(1024), 512, nfft_psd, config.fs);

% Quantitative PSD deltas at key tones (1 kHz target, 4 kHz interference)
[~, idx_1k] = min(abs(Fp - 1000));
[~, idx_4k] = min(abs(Fp - 4000));

psd_in_1k_dB = 10*log10(Pxx_in(idx_1k) + eps);
psd_tgt_1k_dB = 10*log10(Pxx_out_target(idx_1k) + eps);
psd_int_1k_dB = 10*log10(Pxx_out_interf(idx_1k) + eps);

psd_in_4k_dB = 10*log10(Pxx_in(idx_4k) + eps);
psd_tgt_4k_dB = 10*log10(Pxx_out_target(idx_4k) + eps);
psd_int_4k_dB = 10*log10(Pxx_out_interf(idx_4k) + eps);

delta_target_1k_dB = psd_tgt_1k_dB - psd_in_1k_dB;
delta_supp_4k_dB = psd_in_4k_dB - psd_tgt_4k_dB;
delta_int_supp_1k_dB = psd_in_1k_dB - psd_int_1k_dB;
delta_int_keep_4k_dB = psd_int_4k_dB - psd_in_4k_dB;

fprintf('\n===== PSD DELTAS AT KEY FREQUENCIES =====\n');
fprintf('1 kHz (target-steered):  out - in = %+7.2f dB\n', delta_target_1k_dB);
fprintf('4 kHz (target-steered):  in - out = %+7.2f dB (suppression)\n', delta_supp_4k_dB);
fprintf('1 kHz (interf-steered):  in - out = %+7.2f dB (suppression)\n', delta_int_supp_1k_dB);
fprintf('4 kHz (interf-steered):  out - in = %+7.2f dB\n', delta_int_keep_4k_dB);
fprintf('==========================================\n');

results_dir = fullfile(config.output_dir, 'results');
if exist(results_dir, 'dir') ~= 7
    mkdir(results_dir);
end

metrics_table = table( ...
    [Fp(idx_1k); Fp(idx_4k)], ...
    [psd_in_1k_dB; psd_in_4k_dB], ...
    [psd_tgt_1k_dB; psd_tgt_4k_dB], ...
    [psd_int_1k_dB; psd_int_4k_dB], ...
    [delta_target_1k_dB; NaN], ...
    [NaN; delta_supp_4k_dB], ...
    [delta_int_supp_1k_dB; NaN], ...
    [NaN; delta_int_keep_4k_dB], ...
    'VariableNames', {'Frequency_Hz','Input_dBHz','TargetSteered_dBHz','InterfSteered_dBHz', ...
                      'TargetSteered_1k_OutMinusIn_dB','TargetSteered_4k_InMinusOut_dB', ...
                      'InterfSteered_1k_InMinusOut_dB','InterfSteered_4k_OutMinusIn_dB'});

writetable(metrics_table, fullfile(results_dir, 'PSD_key_deltas_1k_4k.csv'));

figure(209); clf;
plot(Fp, 10*log10(Pxx_in + eps), 'k', 'LineWidth', 1.5); hold on;
plot(Fp, 10*log10(Pxx_out_target + eps), 'b', 'LineWidth', 1.5);
plot(Fp, 10*log10(Pxx_out_interf + eps), 'm', 'LineWidth', 1.5);
xlim([0 config.fs/2]);
xlabel('Frequency (Hz)', 'FontSize', font_sz.label); ylabel('PSD (dB/Hz)', 'FontSize', font_sz.label);
h = legend('Input', '2kHz-steered', '4kHz-steered', 'Location', 'best');
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', 'white', 'EdgeColor', 'k', 'FontSize', font_sz.legend);
grid on;
set(gca, 'LineWidth', 1, 'FontSize', font_sz.tick);
save_figure(path_psd_fullband);

figure(210); clf;
plot(Fp, 10*log10(Pxx_in + eps), 'k', 'LineWidth', 1.5); hold on;
plot(Fp, 10*log10(Pxx_out_interf + eps), 'm', 'LineWidth', 1.5);
xlim([3500 4500]);
xlabel('Frequency (Hz)', 'FontSize', font_sz.label); ylabel('PSD (dB/Hz)', 'FontSize', font_sz.label);
h = legend('Input', '4kHz-steered', 'Location', 'best');
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', 'white', 'EdgeColor', 'k', 'FontSize', font_sz.legend);
grid on;
set(gca, 'LineWidth', 1, 'FontSize', font_sz.tick);
save_figure(path_psd_4kHz);

fprintf('[Diagnostics] Visualization complete.\n\n');

end
