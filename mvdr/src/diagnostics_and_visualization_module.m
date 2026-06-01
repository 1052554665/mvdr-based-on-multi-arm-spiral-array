%% ============== Diagnostics and Visualization Module ==============
% Computes diagnostics (SNR improvement, beampatterns) and generates visualizations

function diagnostics_and_visualization_module(config, Yf_tgt, Y_tar_tgt, Y_intnoi_tgt, ...
    Yf_int, Y_interf_int, Y_tarnoi_int, S, S_interf, S_intnoi, S_tarnoi, ...
    X_noisy, X_target, X_interf, F, T, a_target, a_interf, mic_pos, r_center, Nt, Nmic)

fprintf('\n[Diagnostics] Starting diagnostics and visualization...\n');
% Improve visibility: bold black axes/text by default for generated figures
set(groot, 'defaultAxesFontWeight', 'bold', 'defaultAxesFontSize', 12, ...
    'defaultTextColor', 'k', 'defaultAxesXColor', 'k', 'defaultAxesYColor', 'k', 'defaultAxesZColor', 'k');

plot_style = config.plot_style;
font_sz = plot_style.font_sz;
line_w = plot_style.line_width;
marker_sz = plot_style.marker_size;
figure_bg = plot_style.figure_bg;
axes_bg = plot_style.axes_bg;
text_color = plot_style.text_color;
if isfield(plot_style, 'colormap_name')
    cmap_name = plot_style.colormap_name;
else
    cmap_name = 'turbo';
end
figure213_annotations = config.figure213_annotations;

%% Define figure output paths
figures_dir = fullfile(config.output_dir, config.figure_subdir);
path_eigenspectrum_2kHz = fullfile(figures_dir, 'Rxx_eigenspectrum_2kHz.pdf');
path_eigenspectrum_fullband = fullfile(figures_dir, 'Rxx_eigenspectrum_fullband.pdf');
path_beampattern = fullfile(figures_dir, 'Beampattern.pdf');
path_beampattern_polar = fullfile(figures_dir, 'Beampattern_polar.pdf');
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
% Respect per-experiment configuration when present; otherwise use
% repository defaults. This keeps behavior stable while allowing
% experiments to override any subset of font sizes.
default_font_sz.title = 32;          % Main titles for single plots
default_font_sz.subtitle = 31;       % Subtitle/axis labels for single plots
default_font_sz.label = 31;          % Axis labels (xlabel, ylabel)
default_font_sz.subtile_title = 31;  % Titles in subplots (tiledlayout)
default_font_sz.legend = 30;         % Legend text
default_font_sz.tick = 30;           % Tick labels
default_font_sz.small_tick = 29;     % Small tick labels for dense subplots
default_font_sz.colorbar = 30;       % Colorbar tick labels
default_font_sz.colorbar_label = 30; % Colorbar label

if ~isfield(plot_style, 'font_sz') || isempty(plot_style.font_sz)
    font_sz = default_font_sz;
else
    font_sz = plot_style.font_sz;
    % Fill missing fields from defaults
    fn = fieldnames(default_font_sz);
    for ii = 1:numel(fn)
        f = fn{ii};
        if ~isfield(font_sz, f) || isempty(font_sz.(f))
            font_sz.(f) = default_font_sz.(f);
        end
    end
end
%% Figure save helper function
save_figure = @(fig_path) save_fig_func(config, fig_path);
    function save_fig_func(config, fig_path)
        if config.save_figures
            set(gcf,'Color',figure_bg);
            set(gca,'Color',axes_bg);
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
set(gcf, 'Color', figure_bg);
set(gca, 'Color', axes_bg);
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
set(gcf, 'Color', figure_bg);
set(gca, 'Color', axes_bg);
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

fprintf('[Diagnostics] 2kHz eigenspectrum:\n');
fprintf('  Max eigenvalue: %.4e\n', evals(1));
fprintf('  Top 3 eigenvalues: %.4e, %.4e, %.4e\n', evals(1), evals(2), evals(3));

figure(203); clf;
plot(1:Nmic, 10*log10(evals + eps), '-o', 'LineWidth', line_w.medium, 'MarkerSize', marker_sz.main, 'Color', text_color);
xlabel('Index', 'FontSize', font_sz.label); ylabel('Eigenvalue (dB)', 'FontSize', font_sz.label);
% title(sprintf('Covariance Eigenspectrum at %.0f Hz', F(k2k)));
grid on;
set(gca, 'LineWidth', 1.5, 'FontSize', font_sz.tick);
save_figure(path_eigenspectrum_2kHz);

% Full-band eigenspectrum (time-domain mixed signal including both 2kHz and 4kHz)
fprintf('[Diagnostics] Computing full-band eigenspectrum...\n');
t1_full = max(1, round(numFrames*0.35));
t2_full = min(size(X_noisy, 1), round(Nt*0.65));
Xloc_full = X_noisy(t1_full:t2_full, :).';  % (Nmic, Nsamples_windowed)
Rxx_full = (Xloc_full * Xloc_full') / size(Xloc_full, 2);
reg_full = config.epsilon * trace(Rxx_full) / Nmic;
Rxx_full = Rxx_full + reg_full * eye(Nmic);

[~, D_full] = eig(Rxx_full);
evals_full = sort(diag(D_full), 'descend');

fprintf('  Max eigenvalue: %.4e\n', evals_full(1));
fprintf('  Top 3 eigenvalues: %.4e, %.4e, %.4e\n', evals_full(1), evals_full(2), evals_full(3));
fprintf('  Eigenvalue ratio (1st/2nd): %.2f\n', evals_full(1)/max(evals_full(2), eps));

% Full eigenvalue analysis - detect number of signal sources
fprintf('\n  === Full Eigenvalue Spectrum Analysis ===\n');
fprintf('  Eigenvalue Index    Value         Ratio to Max   dB\n');
fprintf('  %-18s %-13s %-14s %s\n', '---', '---', '---', '---');
for i = 1:min(Nmic, length(evals_full))
    ratio_to_max = evals_full(i) / (evals_full(1) + eps);
    db_val = 10*log10(evals_full(i) + eps);
    fprintf('  %-18d %.4e        %.4f        %.2f\n', i, evals_full(i), ratio_to_max, db_val);
end

% Detect number of signal sources using threshold
threshold = 0.01;  % Eigenvalues > 1% of max are signal sources
num_signals = sum(evals_full > threshold * evals_full(1));
fprintf('\n  Estimated # of signal sources (threshold=%.1f%% of max): %d\n', threshold*100, num_signals);
fprintf('  Condition number: %.2e\n', evals_full(1) / (evals_full(end) + eps));

% Additional analysis: Find the "elbow" in the eigenvalue curve
% (where eigenvalues drop significantly)
diffs = diff(evals_full);
rel_diffs = abs(diffs) ./ evals_full(1:end-1);
[max_rel_diff, elbow_idx] = max(rel_diffs);
fprintf('  Largest relative drop at index %d->%d (%.1f%%)\n', elbow_idx, elbow_idx+1, max_rel_diff*100);
fprintf('  Signal subspace dimension (by elbow): %d\n', elbow_idx);

figure(214); clf;
plot(1:Nmic, 10*log10(evals_full + eps), '-o', 'LineWidth', line_w.medium, 'MarkerSize', marker_sz.main, 'Color', 'r');
xlabel('Index', 'FontSize', font_sz.label); ylabel('Eigenvalue (dB)', 'FontSize', font_sz.label);
grid on;
set(gca, 'LineWidth', 1.5, 'FontSize', font_sz.tick);
save_figure(path_eigenspectrum_fullband);

% Comparison plot: Normalized 2kHz vs Fullband eigenspectra
figure(215); clf;
evals_norm = evals / max(evals);
evals_full_norm = evals_full / max(evals_full);
semilogy(1:Nmic, evals_norm, '-o', 'LineWidth', line_w.medium, 'MarkerSize', marker_sz.main, 'Color', 'k', 'DisplayName', config.eigen_plot_names{1}); hold on;
semilogy(1:Nmic, evals_full_norm, '-s', 'LineWidth', line_w.medium, 'MarkerSize', marker_sz.main, 'Color', 'r', 'DisplayName', config.eigen_plot_names{2});
xlabel('Eigenvalue Index', 'FontSize', font_sz.label); ylabel('Normalized Eigenvalue', 'FontSize', font_sz.label);
h = legend('FontSize', font_sz.legend, 'Location', 'best');
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', figure_bg, 'EdgeColor', 'k');
grid on;
set(gca, 'LineWidth', 1.5, 'FontSize', font_sz.tick);
set(gcf, 'Color', figure_bg);
set(gca, 'Color', axes_bg);
path_eigenspectrum_comparison = fullfile(figures_dir, 'Eigenspectrum_comparison_2kHz_vs_fullband.pdf');
save_figure(path_eigenspectrum_comparison);

% Detailed eigenvalue analysis figure
figure(216); clf;
ax1 = subplot(2,1,1);
semilogy(1:min(10, length(evals_full)), evals_full(1:min(10, length(evals_full))), '-o', 'LineWidth', 2.5, 'MarkerSize', 10, 'Color', 'r');
hold on;
yline(0.01 * evals_full(1), '--', 'LineWidth', 1.5, 'Color', 'b', 'DisplayName', '1% threshold');
yline(evals_full(end), '--', 'LineWidth', 1.5, 'Color', 'g', 'DisplayName', 'Noise floor');
xlabel('Eigenvalue Index', 'FontSize', font_sz.label);
ylabel('Eigenvalue', 'FontSize', font_sz.label);
% title('Full-band Eigenvalue Spectrum (Top 10)', 'FontSize', font_sz.title);
h = legend('FontSize', font_sz.legend, 'Location', 'best');
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', figure_bg, 'EdgeColor', 'k');
grid on;
set(ax1, 'LineWidth', 1.5, 'FontSize', font_sz.tick, 'Color', axes_bg, 'XColor', text_color, 'YColor', text_color);

% Relative magnitude plot
ax2 = subplot(2,1,2);
rel_evals = evals_full / evals_full(1) * 100;
bar(1:min(10, length(rel_evals)), rel_evals(1:min(10, length(rel_evals))), 'FaceColor', 'r', 'EdgeColor', 'k', 'LineWidth', 1.5);
hold on;
yline(1, '--', 'LineWidth', 1.5, 'Color', 'b', 'DisplayName', '1% threshold');
xlabel('Eigenvalue Index', 'FontSize', font_sz.label);
ylabel('Relative Magnitude (%)', 'FontSize', font_sz.label);
% title('Relative Eigenvalue Magnitudes', 'FontSize', font_sz.title);
h = legend('FontSize', font_sz.legend, 'Location', 'best');
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', figure_bg, 'EdgeColor', 'k');
set(ax2, 'LineWidth', 1.5, 'FontSize', font_sz.tick, 'YScale', 'log', 'Color', axes_bg, 'XColor', text_color, 'YColor', text_color);
grid on;

set(gcf, 'Color', figure_bg);
path_eigenanalysis = fullfile(figures_dir, 'Eigenvalue_detailed_analysis.pdf');
save_figure(path_eigenanalysis);

% Beampattern
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

% -3 dB beamwidth around the main lobe
[resp_peak, idx_peak] = max(resp_2k);
bw_level = resp_peak - 3;
N_az = numel(azs);
azs_ext = [azs - 360, azs, azs + 360];
resp_ext = [resp_2k, resp_2k, resp_2k];
idx_peak_ext = idx_peak + N_az;

left_idx = idx_peak_ext;
while left_idx > 1 && resp_ext(left_idx) >= bw_level && (idx_peak_ext - left_idx) <= N_az
    left_idx = left_idx - 1;
end

right_idx = idx_peak_ext;
while right_idx < numel(resp_ext) && resp_ext(right_idx) >= bw_level && (right_idx - idx_peak_ext) <= N_az
    right_idx = right_idx + 1;
end

left_edge_az = azs_ext(min(left_idx + 1, numel(azs_ext)));
right_edge_az = azs_ext(max(right_idx - 1, 1));
beamwidth_3db = right_edge_az - left_edge_az;

left_edge_az_plot = mod(left_edge_az + 180, 360) - 180;
right_edge_az_plot = mod(right_edge_az + 180, 360) - 180;
peak_az_plot = mod(azs(idx_peak) + 180, 360) - 180;

figure(204); clf;
plot(azs, resp_2k, 'LineWidth', line_w.medium, 'Color', 'k'); hold on;
yline(bw_level, '--', 'LineWidth', line_w.main, 'Color', [0.85 0.2 0.2]);
y_lim = ylim;
plot([left_edge_az_plot left_edge_az_plot], y_lim, '--', 'LineWidth', line_w.main, 'Color', [0.85 0.2 0.2]);
plot([right_edge_az_plot right_edge_az_plot], y_lim, '--', 'LineWidth', line_w.main, 'Color', [0.85 0.2 0.2]);
plot(peak_az_plot, resp_peak, 'o', 'MarkerSize', marker_sz.main, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
text(peak_az_plot, resp_peak - 2, sprintf('BW_{-3dB}=%.1f^o', beamwidth_3db), ...
    'FontSize', font_sz.small_tick, 'FontWeight', 'bold', 'Color', text_color, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
xlabel('Azimuth (deg)', 'FontSize', font_sz.label); ylabel('Response (dB)', 'FontSize', font_sz.label);
% title(sprintf('Target-Steered Beampattern at %.0f Hz', F(k2k)));
grid on;
set(gca, 'LineWidth', 1.5, 'FontSize', font_sz.tick);
save_figure(path_beampattern);

% Polar-coordinate beampattern
figure(217); clf;
resp_2k_norm = resp_2k - resp_peak;
resp_2k_norm = max(resp_2k_norm, -60);
polarplot(deg2rad(azs), resp_2k_norm, 'k', 'LineWidth', line_w.medium); hold on;
polarplot(deg2rad([left_edge_az_plot left_edge_az_plot]), [-60 0], '--', 'LineWidth', line_w.main, 'Color', [0.85 0.2 0.2]);
polarplot(deg2rad([right_edge_az_plot right_edge_az_plot]), [-60 0], '--', 'LineWidth', line_w.main, 'Color', [0.85 0.2 0.2]);
polarplot(deg2rad(peak_az_plot), 0, 'o', 'MarkerSize', marker_sz.main, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
axp = gca;
axp.ThetaZeroLocation = 'top';
axp.ThetaDir = 'clockwise';
axp.RLim = [-60 0];
axp.RTick = -60:10:0;
axp.FontSize = font_sz.small_tick;
% title(sprintf('2 kHz Beampattern (Polar), BW_{-3dB}=%.1f^o', beamwidth_3db), 'FontSize', font_sz.subtitle, 'Color', text_color);
save_figure(path_beampattern_polar);

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
if isfield(config, 'legend_mvdr') && numel(config.legend_mvdr) >= 2
    h = legend(config.legend_mvdr{1}, config.legend_mvdr{2}, 'Location', 'best');
else
    h = legend('Target-steered', 'Interference-steered', 'Location', 'best');
end
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', figure_bg, 'EdgeColor', 'k', 'FontSize', font_sz.legend);
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
% t = title('Input Target (Mic 1)'); set(t, 'Color', 'k', 'FontName', 'Times New Roman'); 
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

nexttile;
plot(t_cmp, x_interf_cmp, 'm', 'LineWidth', 1.1);
% t = title('Input Interference (Mic 1)'); set(t, 'Color', 'k', 'FontName', 'Times New Roman');

xlabel('Time (s)'); ylabel('Amplitude'); grid on;

nexttile;
plot(t_cmp, x_noisy_cmp, 'k', 'LineWidth', 1.1);
% t = title('Input Received (Mic 1)'); set(t, 'Color', 'k', 'FontName', 'Times New Roman');

xlabel('Time (s)'); ylabel('Amplitude'); grid on;

nexttile;
plot(t_cmp, y_target_cmp, 'b', 'LineWidth', 1.1);
% t = title('MVDR Processed Target'); set(t, 'Color', 'k', 'FontName', 'Times New Roman');

xlabel('Time (s)'); ylabel('Amplitude'); grid on;

nexttile;
plot(t_cmp, y_interf_cmp, 'm', 'LineWidth', 1.1);
% t = title('MVDR Processed Interference'); set(t, 'Color', 'k', 'FontName', 'Times New Roman'); 
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

nexttile;
plot(t_cmp, y_recv_cmp, 'k', 'LineWidth', 1.1);
% t = title('MVDR Processed Received'); set(t, 'Color', 'k', 'FontName', 'Times New Roman'); 

xlabel('Time (s)'); ylabel('Amplitude'); grid on;

% sg = sgtitle('Input vs MVDR-Processed Signals');
% set(sg, 'Color', 'k', 'FontWeight', 'bold', 'FontName', 'Times New Roman', 'FontSize', 20);
set(findall(gcf,'type','axes'), 'LineWidth', 1, 'FontSize', font_sz.small_tick);
set(gcf, 'Color', figure_bg);
set(findall(gcf,'type','axes'), 'Color', axes_bg, 'XColor', text_color, 'YColor', text_color);
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
% t = title('Input Target (Mic 1)'); set(t, 'Color', 'k', 'FontSize', font_sz.subtile_title);
xlabel('Time (s)', 'FontSize', font_sz.subtitle); ylabel('Frequency (Hz)', 'FontSize', font_sz.subtitle);
ylim([0 config.fs/2]); caxis([spec_clim_min spec_clim_max]); set(gca, 'FontSize', font_sz.small_tick);

nexttile;
imagesc(Tspec, Fspec, P_xi); axis xy;
% t = title('Input Interference (Mic 1)'); set(t, 'Color', 'k', 'FontSize', font_sz.subtile_title); 
xlabel('Time (s)', 'FontSize', font_sz.subtitle); ylabel('Frequency (Hz)', 'FontSize', font_sz.subtitle);
ylim([0 config.fs/2]); caxis([spec_clim_min spec_clim_max]); set(gca, 'FontSize', font_sz.small_tick);

nexttile;
imagesc(Tspec, Fspec, P_xr); axis xy;
% t = title('Input Received (Mic 1)'); set(t, 'Color', 'k', 'FontSize', font_sz.subtile_title);
xlabel('Time (s)', 'FontSize', font_sz.subtitle); ylabel('Frequency (Hz)', 'FontSize', font_sz.subtitle);
ylim([0 config.fs/2]); caxis([spec_clim_min spec_clim_max]); set(gca, 'FontSize', font_sz.small_tick);

nexttile;
imagesc(Tspec, Fspec, P_yt); axis xy;
% t = title('MVDR Processed Target'); set(t, 'Color', 'k', 'FontSize', font_sz.subtile_title); 
xlabel('Time (s)', 'FontSize', font_sz.subtitle); ylabel('Frequency (Hz)', 'FontSize', font_sz.subtitle);
ylim([0 config.fs/2]); caxis([spec_clim_min spec_clim_max]); set(gca, 'FontSize', font_sz.small_tick);

nexttile;
imagesc(Tspec, Fspec, P_yi); axis xy;
% t = title('MVDR Processed Interference'); set(t, 'Color', 'k', 'FontSize', font_sz.subtile_title); 
xlabel('Time (s)', 'FontSize', font_sz.subtitle); ylabel('Frequency (Hz)', 'FontSize', font_sz.subtitle);
ylim([0 config.fs/2]); caxis([spec_clim_min spec_clim_max]); set(gca, 'FontSize', font_sz.small_tick);

nexttile;
imagesc(Tspec, Fspec, P_yr); axis xy;
% t = title('MVDR Processed Received'); set(t, 'Color', 'k', 'FontSize', font_sz.subtile_title);

xlabel('Time (s)', 'FontSize', font_sz.subtitle); ylabel('Frequency (Hz)', 'FontSize', font_sz.subtitle);
ylim([0 config.fs/2]); caxis([spec_clim_min spec_clim_max]); set(gca, 'FontSize', font_sz.small_tick);

colormap(cmap_name);
set(findall(gcf,'type','axes'), 'LineWidth', 1, 'Color', axes_bg, 'XColor', text_color, 'YColor', text_color, 'FontSize', font_sz.small_tick);
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
set(gcf, 'Color', figure_bg);
save_figure(path_spectrograms);

%% 6a. MVDR Interference Suppression Comparison (3-tile highlight)
fprintf('[Diagnostics] Generating MVDR interference suppression comparison...\n');

figure(213); clf;
tiledlayout(1,3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
imagesc(Tspec, Fspec, P_xr); axis xy;
% t = title('Input Received (Mixed)', 'FontWeight', 'bold');
% set(t, 'Color', 'k', 'FontSize', font_sz.subtile_title);
xlabel('Time (s)', 'FontSize', font_sz.subtitle);
ylabel('Frequency (Hz)', 'FontSize', font_sz.subtitle);
ylim([0 config.fs/2]);
caxis([spec_clim_min spec_clim_max]);
set(gca, 'FontSize', font_sz.small_tick);
text(0.02, 0.95, figure213_annotations.mixed.text, 'Units', 'normalized', 'FontSize', figure213_annotations.mixed.font_size, ...
    'Color', figure213_annotations.mixed.color, 'FontWeight', figure213_annotations.mixed.font_weight, ...
    'VerticalAlignment', 'top', 'BackgroundColor', figure213_annotations.mixed.background_color);

nexttile;
imagesc(Tspec, Fspec, P_yt); axis xy;
% t = title('Target-Steered MVDR', 'FontWeight', 'bold');
% set(t, 'Color', 'k', 'FontSize', font_sz.subtile_title);
xlabel('Time (s)', 'FontSize', font_sz.subtitle);
ylabel('Frequency (Hz)', 'FontSize', font_sz.subtitle);
ylim([0 config.fs/2]);
caxis([spec_clim_min spec_clim_max]);
set(gca, 'FontSize', font_sz.small_tick);
text(0.02, 0.95, figure213_annotations.target.text, 'Units', 'normalized', 'FontSize', figure213_annotations.target.font_size, ...
    'Color', figure213_annotations.target.color, 'FontWeight', figure213_annotations.target.font_weight, ...
    'VerticalAlignment', 'top', 'BackgroundColor', figure213_annotations.target.background_color);

nexttile;
imagesc(Tspec, Fspec, P_yi); axis xy;
% t = title('Interference-Steered MVDR', 'FontWeight', 'bold');
% set(t, 'Color', 'k', 'FontSize', font_sz.subtile_title);
xlabel('Time (s)', 'FontSize', font_sz.subtitle);
ylabel('Frequency (Hz)', 'FontSize', font_sz.subtitle);
ylim([0 config.fs/2]);
caxis([spec_clim_min spec_clim_max]);
set(gca, 'FontSize', font_sz.small_tick);
text(0.02, 0.95, figure213_annotations.interference.text, 'Units', 'normalized', 'FontSize', figure213_annotations.interference.font_size, ...
    'Color', figure213_annotations.interference.color, 'FontWeight', figure213_annotations.interference.font_weight, ...
    'VerticalAlignment', 'top', 'BackgroundColor', figure213_annotations.interference.background_color);

colormap(cmap_name);
set(findall(gcf,'type','axes'), 'LineWidth', 1, 'Color', axes_bg, 'XColor', text_color, 'YColor', text_color, 'FontSize', font_sz.small_tick);
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Power/Frequency (dB)';
cb.Color = 'k';
cb.FontWeight = 'bold';
cb.Label.Color = 'k';
cb.Label.FontWeight = 'bold';
cb.FontSize = font_sz.colorbar;
cb.Label.FontSize = font_sz.colorbar_label;
set(gcf, 'Color', figure_bg);

path_mvdr_suppression = fullfile(figures_dir, 'MVDR_interference_suppression_comparison.pdf');
save_figure(path_mvdr_suppression);

%% 7. Performance bar chart
fprintf('[Diagnostics] Generating performance chart...\n');

figure(208); clf;
bar([SNR_in_dB, SNR_out_dB; ISR_in_dB, ISR_out_dB]);
set(gca, 'XTickLabel', {'2kHz-steered', '4kHz-steered'}, 'FontSize', font_sz.tick);
ylabel('Ratio (dB)', 'FontSize', font_sz.label);
h = legend('Before MVDR', 'After MVDR', 'Location', 'best');
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', figure_bg, 'EdgeColor', 'k', 'FontSize', font_sz.legend);
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
% h = legend('Input', '2kHz-steered', '4kHz-steered', 'Location', 'best');
h = legend('Target-steered', 'Interference-steered', 'Location', 'best');
if isfield(config, 'legend_psd') && numel(config.legend_psd) >= 2
    h = legend(config.legend_psd{1}, config.legend_psd{2}, 'Location', 'best');
end
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', figure_bg, 'EdgeColor', 'k', 'FontSize', font_sz.legend);
grid on;
set(gca, 'LineWidth', 1, 'FontSize', font_sz.tick);
save_figure(path_psd_fullband);

figure(210); clf;
plot(Fp, 10*log10(Pxx_in + eps), 'k', 'LineWidth', 1.5); hold on;
plot(Fp, 10*log10(Pxx_out_interf + eps), 'm', 'LineWidth', 1.5);
xlim([3500 4500]);
xlabel('Frequency (Hz)', 'FontSize', font_sz.label); ylabel('PSD (dB/Hz)', 'FontSize', font_sz.label);
h = legend('Input', '4kHz-steered', 'Location', 'best');
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', figure_bg, 'EdgeColor', 'k', 'FontSize', font_sz.legend);
grid on;
set(gca, 'LineWidth', 1, 'FontSize', font_sz.tick);
save_figure(path_psd_4kHz);

fprintf('[Diagnostics] Visualization complete.\n\n');

end
