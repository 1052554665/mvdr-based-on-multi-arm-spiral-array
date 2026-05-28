%% ============================================================
%% MVDR Multi-Beta Parameter Sweep Analysis
%% ============================================================
%% Analyzes MVDR performance across varying wall reflection coefficients
%% to demonstrate the impact of room acoustics on beamforming
%% ============================================================

clear; close all; clc;

%% Add paths
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
external_rir_dir = fullfile(fileparts(project_root), 'RIR-Generator');

cd(script_dir);
addpath(genpath(project_root));
if exist(external_rir_dir, 'dir') == 7
    addpath(genpath(external_rir_dir));
else
    warning('MVDR:Path', 'RIR-Generator folder not found');
end

fprintf('====================================================\n');
fprintf('  MVDR Multi-Beta Analysis (Room Reflection Effects)\n');
fprintf('====================================================\n\n');

%% Define parameter sweep (wall reflection coefficients)
beta_values = [0, 0.2, 0.4, 0.6, 0.8];
n_betas = length(beta_values);
fprintf('[Analysis] Testing %d beta values: %s\n', n_betas, mat2str(beta_values));

%% Storage for results
evals_all = cell(n_betas, 1);    % Eigenvalues for each beta
evals_full_all = cell(n_betas, 1);
SNR_gains = zeros(n_betas, 1);
ISR_gains = zeros(n_betas, 1);
cond_numbers = zeros(n_betas, 1);

%% Load baseline configuration
fprintf('[Analysis] Loading base configuration...\n');
config = load_config('real_signal_dcbias_4k');
% config = load_config('demo_2k_4k');


%% Create output directory structure
base_output_dir = config.output_dir;
if ~isfolder(base_output_dir)
    mkdir(base_output_dir);
end

%% Loop over beta values
for b_idx = 1:n_betas
    beta_val = beta_values(b_idx);
    fprintf('\n========== Beta = %.1f (%d/%d) ==========\n', beta_val, b_idx, n_betas);

    % Set beta parameter
    config.beta = beta_val;

    %% Load microphone positions and audio signals
    [mic_pos, X_target, X_interf, X_noisy, x_target, x_interf, t, Nt, Nmic, r_center] = ...
        load_data_module(config);

    %% Generate RIRs with current beta
    [h_target, h_interf, X_target, X_interf, X_noisy] = ...
        rir_generation_module(config, x_target, x_interf, mic_pos, Nt, Nmic);

    %% MVDR processing
    [Yf_tgt, Y_tar_tgt, Y_intnoi_tgt, Yf_int, Y_interf_int, Y_tarnoi_int, ...
     S, S_tar, S_interf, S_intnoi, S_tarnoi, F, T, a_target, a_interf] = ...
        mvdr_processing_module(config, X_noisy, X_target, X_interf, mic_pos, r_center, Nt, Nmic);

    %% Extract eigenvalue information at 2 kHz
    numFrames = size(Yf_tgt, 2);
    [~, k2k] = min(abs(F - 2000));

    t1_diag = max(1, round(numFrames*0.35));
    t2_diag = min(numFrames, round(numFrames*0.65));

    Xkf_k2 = squeeze(S(k2k,:,:)).';
    t2_diag = min(t2_diag, size(Xkf_k2,2));
    Xloc_diag = Xkf_k2(:, t1_diag:t2_diag);
    Rxx_diag = (Xloc_diag * Xloc_diag') / size(Xloc_diag,2);
    reg_diag = config.epsilon * trace(Rxx_diag) / Nmic;
    Rxx_diag = Rxx_diag + reg_diag * eye(Nmic);

    [~, D] = eig(Rxx_diag);
    evals_all{b_idx} = sort(diag(D), 'descend');

    %% Full-band eigenspectrum
    t1_full = max(1, round(numFrames*0.35));
    t2_full = min(size(X_noisy, 1), round(Nt*0.65));
    Xloc_full = X_noisy(t1_full:t2_full, :).';
    Rxx_full = (Xloc_full * Xloc_full') / size(Xloc_full, 2);
    reg_full = config.epsilon * trace(Rxx_full) / Nmic;
    Rxx_full = Rxx_full + reg_full * eye(Nmic);

    [~, D_full] = eig(Rxx_full);
    evals_full_all{b_idx} = sort(diag(D_full), 'descend');
    cond_numbers(b_idx) = evals_full_all{b_idx}(1) / max(evals_full_all{b_idx}(end), eps);

    %% Compute performance metrics
    win = hamming(config.win_len);
    y_mvdr_target = real(istft(Yf_tgt, config.fs, ...
        'Window', win, 'OverlapLength', config.noverlap, 'FFTLength', config.Nfft));
    y_tar_out_target = real(istft(Y_tar_tgt, config.fs, ...
        'Window', win, 'OverlapLength', config.noverlap, 'FFTLength', config.Nfft));
    y_intnoi_out_target = real(istft(Y_intnoi_tgt, config.fs, ...
        'Window', win, 'OverlapLength', config.noverlap, 'FFTLength', config.Nfft));

    y_mvdr_interf = real(istft(Yf_int, config.fs, ...
        'Window', win, 'OverlapLength', config.noverlap, 'FFTLength', config.Nfft));
    y_interf_out_interf = real(istft(Y_interf_int, config.fs, ...
        'Window', win, 'OverlapLength', config.noverlap, 'FFTLength', config.Nfft));
    y_tarnoi_out_interf = real(istft(Y_tarnoi_int, config.fs, ...
        'Window', win, 'OverlapLength', config.noverlap, 'FFTLength', config.Nfft));

    % Target-steered metrics
    P_tar_out = mean(y_tar_out_target.^2);
    P_intnoi_out = mean(y_intnoi_out_target.^2);
    SNR_out_dB = 10*log10(P_tar_out / (P_intnoi_out + eps));

    x_tar_in = X_target(:, config.ref_mic);
    P_tar_in = mean(x_tar_in.^2);
    x_int_in = X_interf(:, config.ref_mic);
    P_int_in = mean(x_int_in.^2);
    SNR_in_dB = 10*log10(P_tar_in / (P_int_in + eps));

    SNR_gains(b_idx) = SNR_out_dB - SNR_in_dB;

    % Interference-steered metrics
    P_interf_in = mean(X_interf(:, config.ref_mic).^2);
    P_tarnoi_in = mean(X_target(:, config.ref_mic).^2);
    ISR_in_dB = 10*log10(P_interf_in / (P_tarnoi_in + eps));

    P_interf_out = mean(y_interf_out_interf.^2);
    P_tarnoi_out = mean(y_tarnoi_out_interf.^2);
    ISR_out_dB = 10*log10(P_interf_out / (P_tarnoi_out + eps));
    ISR_gains(b_idx) = ISR_out_dB - ISR_in_dB;

    fprintf('[Beta=%.1f] SNR Gain: %.2f dB | ISR Gain: %.2f dB | Cond: %.2e\n', ...
        beta_val, SNR_gains(b_idx), ISR_gains(b_idx), cond_numbers(b_idx));

end

fprintf('\n====== Multi-Beta Analysis Complete ======\n');

%% Generate comparison figures
output_dir = fullfile(base_output_dir, 'multi_beta_comparison');
if ~isfolder(output_dir)
    mkdir(output_dir);
end

% Font settings (IEEE style)
set(groot, 'defaultAxesFontName','Times New Roman', 'defaultTextFontName','Times New Roman', ...
    'defaultLineLineWidth', 1.2, 'defaultAxesFontSize', 14, 'defaultAxesLineWidth', 1.5);

font_sz.title = 24;
font_sz.label = 22;
font_sz.legend = 20;
font_sz.tick = 19;

%% Figure 1: Eigenvalue Spectra Comparison (2 kHz)
figure(301); clf;
% IEEE-standard color palette (print and B/W friendly)
ieee_colors = [
    0.0   0.0   0.0      % β=0.0: Black
    0.0   0.447 0.741    % β=0.2: Blue
    0.85  0.325 0.098    % β=0.4: Red/Orange
    0.466 0.674 0.188    % β=0.6: Green
    0.75  0.0   0.75     % β=0.8: Purple
];
line_styles = {'-', '-', '--', '-.', ':'};
markers = {'o', 's', '^', 'd', 'v'};

for b_idx = 1:n_betas
    evals = evals_all{b_idx};
    Nmic = length(evals);
    semilogy(1:Nmic, evals + eps, 'LineWidth', 2.2, 'MarkerSize', 7, ...
        'Color', ieee_colors(b_idx,:), 'LineStyle', line_styles{b_idx}, 'Marker', markers{b_idx}, ...
        'DisplayName', sprintf('\\beta = %.1f', beta_values(b_idx)), 'MarkerFaceColor', 'none');
    hold on;
end

xlabel('Eigenvalue Index', 'FontSize', font_sz.label);
ylabel('Eigenvalue Magnitude', 'FontSize', font_sz.label);
% title('Covariance Eigenspectra at 2 kHz vs. Reflection Coefficient', 'FontSize', font_sz.title);
h = legend('FontSize', font_sz.legend, 'Location', 'best');
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', 'white', 'EdgeColor', 'k');
grid on; set(gca, 'FontSize', font_sz.tick, 'LineWidth', 1.5);
set(gcf, 'Color', 'white'); set(gca, 'Color', 'white');

exportgraphics(gcf, fullfile(output_dir, 'Eigenspectra_2kHz_comparison.pdf'), ...
    'ContentType','vector', 'BackgroundColor','white', 'Resolution', 600);

%% Figure 2: Full-Band Eigenvalue Spectra Comparison
figure(302); clf;
% IEEE-standard color palette (consistent with Fig. 1)
ieee_colors = [
    0.0   0.0   0.0      % β=0.0: Black
    0.0   0.447 0.741    % β=0.2: Blue
    0.85  0.325 0.098    % β=0.4: Red/Orange
    0.466 0.674 0.188    % β=0.6: Green
    0.75  0.0   0.75     % β=0.8: Purple
];
line_styles = {'-', '-', '--', '-.', ':'};
markers = {'o', 's', '^', 'd', 'v'};

for b_idx = 1:n_betas
    evals_full = evals_full_all{b_idx};
    Nmic = length(evals_full);
    semilogy(1:Nmic, evals_full + eps, 'LineWidth', 2.2, 'MarkerSize', 7, ...
        'Color', ieee_colors(b_idx,:), 'LineStyle', line_styles{b_idx}, 'Marker', markers{b_idx}, ...
        'DisplayName', sprintf('\\beta = %.1f', beta_values(b_idx)), 'MarkerFaceColor', 'none');
    hold on;
end

xlabel('Eigenvalue Index', 'FontSize', font_sz.label);
ylabel('Eigenvalue Magnitude', 'FontSize', font_sz.label);
% title('Full-Band Covariance Eigenspectra (2 kHz + 4 kHz) vs. Reflection Coefficient', 'FontSize', font_sz.title);
h = legend('FontSize', font_sz.legend, 'Location', 'best');
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', 'white', 'EdgeColor', 'k');
grid on; set(gca, 'FontSize', font_sz.tick, 'LineWidth', 1.5);
set(gcf, 'Color', 'white'); set(gca, 'Color', 'white');

exportgraphics(gcf, fullfile(output_dir, 'Eigenspectra_fullband_comparison.pdf'), ...
    'ContentType','vector', 'BackgroundColor','white', 'Resolution', 600);

%% Figure 3: Comprehensive Performance Metrics vs. Beta (3-in-1)
figure(303); clf;
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% SNR Gain
nexttile;
plot(beta_values, SNR_gains, '-o', 'LineWidth', 2.2, 'MarkerSize', 10, 'Color', 'b');
hold on; grid on;
xlabel('Wall Reflection Coefficient (β)', 'FontSize', font_sz.label);
ylabel('SNR Gain (dB)', 'FontSize', font_sz.label);
% title('Target-Steered: SNR Gain', 'FontSize', font_sz.title);
set(gca, 'FontSize', font_sz.tick, 'LineWidth', 1.5);

% ISR Gain
nexttile;
plot(beta_values, ISR_gains, '-s', 'LineWidth', 2.2, 'MarkerSize', 10, 'Color', 'r');
hold on; grid on;
xlabel('Wall Reflection Coefficient (β)', 'FontSize', font_sz.label);
ylabel('ISR Gain (dB)', 'FontSize', font_sz.label);
% title('Interference-Steered: ISR Gain', 'FontSize', font_sz.title);
set(gca, 'FontSize', font_sz.tick, 'LineWidth', 1.5);

% Condition Number
nexttile;
semilogy(beta_values, cond_numbers, '-^', 'LineWidth', 2.2, 'MarkerSize', 10, 'Color', [0.5 0 0.5]);
hold on; grid on;
xlabel('Wall Reflection Coefficient (β)', 'FontSize', font_sz.label);
ylabel('Condition Number κ', 'FontSize', font_sz.label);
% title('Covariance Matrix Conditioning', 'FontSize', font_sz.title);
set(gca, 'FontSize', font_sz.tick, 'LineWidth', 1.5);

set(gcf, 'Color', 'white'); set(findall(gcf,'type','axes'), 'Color', 'white');

exportgraphics(gcf, fullfile(output_dir, 'Performance_metrics_vs_beta.pdf'), ...
    'ContentType','vector', 'BackgroundColor','white', 'Resolution', 600);

%% (Removed separate Figure 4 - now combined with Figure 3)

%% Generate summary table and report
fprintf('\n===== MULTI-BETA ANALYSIS SUMMARY =====\n');
fprintf('β      SNR Gain (dB)  ISR Gain (dB)  Condition #\n');
fprintf('%.1f     %.2f          %.2f           %.2e\n', ...
    [beta_values; SNR_gains'; ISR_gains'; cond_numbers']);

results_table = table(beta_values(:), SNR_gains(:), ISR_gains(:), cond_numbers(:), ...
    'VariableNames', {'Beta', 'SNR_Gain_dB', 'ISR_Gain_dB', 'Condition_Number'});

writetable(results_table, fullfile(output_dir, 'multi_beta_analysis_results.csv'));

fprintf('\n[Analysis] Comparison figures saved to: %s\n', output_dir);
fprintf('====== Analysis Complete ======\n');
