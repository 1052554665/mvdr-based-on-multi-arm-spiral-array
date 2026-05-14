%% ============== Diagnostics and Visualization Module ==============
% Computes diagnostics (SNR improvement, beampatterns) and generates visualizations

function diagnostics_and_visualization_module(config, Yf_tgt, Y_tar_tgt, Y_intnoi_tgt, ...
    Yf_int, Y_interf_int, Y_tarnoi_int, S, S_interf, S_intnoi, S_tarnoi, ...
    X_noisy, X_target, X_interf, F, T, a_target, a_interf, mic_pos, r_center, Nt, Nmic)

fprintf('\n[Diagnostics] Starting diagnostics and visualization...\n');
% Improve visibility: bold black axes/text by default for generated figures
set(groot, 'defaultAxesFontWeight', 'bold', 'defaultAxesFontSize', 14, ...
    'defaultTextColor', 'k', 'defaultAxesXColor', 'k', 'defaultAxesYColor', 'k', 'defaultAxesZColor', 'k');

win = hamming(config.win_len);
ref_mic = config.ref_mic;

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
plot(1:Nmic, 10*log10(evals + eps), '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Index'); ylabel('Eigenvalue (dB)');
title(sprintf('Covariance Eigenspectrum at %.0f Hz', F(k2k)));
grid on;
set(gca, 'LineWidth', 1.5);
if config.save_figures
    set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
    exportgraphics(gcf, fullfile(config.output_dir, 'figures', 'Rxx_eigenspectrum_2kHz.pdf'), ...
    'ContentType','vector', 'BackgroundColor','white');
end

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
plot(azs, resp_2k, 'LineWidth', 2);
xlabel('Azimuth (deg)'); ylabel('Response (dB)');
title(sprintf('Target-Steered Beampattern at %.0f Hz', F(k2k)));
grid on;
set(gca, 'LineWidth', 1.5);
if config.save_figures
    set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
    exportgraphics(gcf, fullfile(config.output_dir, 'figures', 'Beampattern_2kHz.pdf'), ...
    'ContentType','vector', 'BackgroundColor','white');
end

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
xlabel('Time (s)'); ylabel('Amplitude');
title('Target Signal (Mic 1)');
grid on;
set(gca, 'LineWidth', 1);
if config.save_figures
    set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
    exportgraphics(gcf, fullfile(config.output_dir, 'figures', 'Target_signal.pdf'), 'ContentType','vector', 'BackgroundColor','white');
end

figure(206); clf;
plot(t_axis, X_noisy(:,1), 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Amplitude');
title('Received Signal with Interference (Mic 1)');
grid on;
set(gca, 'LineWidth', 1);
if config.save_figures
    set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
    exportgraphics(gcf, fullfile(config.output_dir, 'figures', 'Received_signal.pdf'), 'ContentType','vector', 'BackgroundColor','white');
end

figure(207); clf;
t_mvdr = (0:length(y_mvdr_target)-1) / config.fs;
plot(t_mvdr, y_mvdr_target, 'b', 'LineWidth', 1.5); hold on;
plot(t_mvdr, y_mvdr_interf, 'm', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Amplitude');
h = legend('Target-steered', 'Interference-steered', 'Location', 'best');
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', 'white', 'EdgeColor', 'k');
title('MVDR Beamformer Outputs');
grid on;
set(gca, 'LineWidth', 1);
if config.save_figures
    set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
    exportgraphics(gcf, fullfile(config.output_dir, 'figures', 'MVDR_outputs.pdf'), 'ContentType','vector', 'BackgroundColor','white');
end

%% 7. Performance bar chart
fprintf('[Diagnostics] Generating performance chart...\n');

figure(208); clf;
bar([SNR_in_dB, SNR_out_dB; ISR_in_dB, ISR_out_dB]);
set(gca, 'XTickLabel', {'Target-steered', 'Interference-steered'});
ylabel('Ratio (dB)');
h = legend('Before MVDR', 'After MVDR', 'Location', 'best');
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', 'white', 'EdgeColor', 'k');
grid on;
set(gca, 'LineWidth', 1);
if config.save_figures
    set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
    exportgraphics(gcf, fullfile(config.output_dir, 'figures', 'Performance_improvement.pdf'), ...
    'ContentType','vector', 'BackgroundColor','white');
end

%% 8. PSD comparison
fprintf('[Diagnostics] Computing power spectral densities...\n');

nfft_psd = 4096;
[Pxx_in, Fp] = pwelch(X_noisy(:,1), hann(1024), 512, nfft_psd, config.fs);
[Pxx_out_target, ~] = pwelch(y_mvdr_target, hann(1024), 512, nfft_psd, config.fs);
[Pxx_out_interf, ~] = pwelch(y_mvdr_interf, hann(1024), 512, nfft_psd, config.fs);

figure(209); clf;
plot(Fp, 10*log10(Pxx_in + eps), 'k', 'LineWidth', 1.5); hold on;
plot(Fp, 10*log10(Pxx_out_target + eps), 'b', 'LineWidth', 1.5);
plot(Fp, 10*log10(Pxx_out_interf + eps), 'm', 'LineWidth', 1.5);
xlim([0 config.fs/2]);
xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
h = legend('Input', 'Target-steered', 'Interference-steered', 'Location', 'best');
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', 'white', 'EdgeColor', 'k');
grid on;
set(gca, 'LineWidth', 1);
if config.save_figures
    set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
    exportgraphics(gcf, fullfile(config.output_dir, 'figures', 'PSD_fullband.pdf'), 'ContentType','vector', 'BackgroundColor','white');
end

figure(210); clf;
plot(Fp, 10*log10(Pxx_in + eps), 'k', 'LineWidth', 1.5); hold on;
plot(Fp, 10*log10(Pxx_out_target + eps), 'b', 'LineWidth', 1.5);
plot(Fp, 10*log10(Pxx_out_interf + eps), 'm', 'LineWidth', 1.5);
xlim([3500 4500]);
xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
h = legend('Input', 'Target-steered', 'Interference-steered', 'Location', 'best');
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', 'white', 'EdgeColor', 'k');
title('PSD around 4 kHz (Interference)');
grid on;
set(gca, 'LineWidth', 1);
if config.save_figures
    set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
    exportgraphics(gcf, fullfile(config.output_dir, 'figures', 'PSD_4kHz.pdf'), 'ContentType','vector', 'BackgroundColor','white');
end

fprintf('[Diagnostics] Visualization complete.\n\n');

end
