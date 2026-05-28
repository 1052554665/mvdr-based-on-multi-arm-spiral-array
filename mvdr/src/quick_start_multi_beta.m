%% ============================================================
%% Quick Start: Multi-Beta MVDR Analysis
%% ============================================================
%% Run this script directly to perform multi-beta analysis
%% with customizable parameters via simple variable assignment
%% ============================================================

%% CONFIGURATION: Modify these parameters before running
% =========================================================

% Define which beta values to test (wall reflection coefficients)
% Typical range: [0, 0.2, 0.4, 0.6, 0.8]
% Lower values = anechoic; Higher values = more reflective
beta_sweep = [0, 0.3, 0.6, 0.9];

% Frequency of analysis (Hz) - typically 2000 for target signal
analysis_freq_hz = 2000;

% Output figure format: 'pdf' or 'png'
output_format = 'pdf';

% Enable GPU acceleration (if available)
use_gpu_accel = true;

% Verbose mode (true = detailed console output)
verbose_mode = true;

% =========================================================
%% Execute analysis
% =========================================================

fprintf('\n╔════════════════════════════════════════════════════════╗\n');
fprintf('║  MVDR Multi-Beta Eigenvalue Analysis Execution        ║\n');
fprintf('║  IEEE Publication-Quality Output                      ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

fprintf('Configuration:\n');
fprintf('  β values: %s\n', mat2str(beta_sweep));
fprintf('  Analysis frequency: %.0f Hz\n', analysis_freq_hz);
fprintf('  GPU acceleration: %s\n', onoff_str(use_gpu_accel));
fprintf('  Output format: %s\n\n', upper(output_format));

%% Add paths and load base config
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
external_rir_dir = fullfile(fileparts(project_root), 'RIR-Generator');

cd(script_dir);
addpath(genpath(project_root));
if exist(external_rir_dir, 'dir') == 7
    addpath(genpath(external_rir_dir));
else
    warning('RIR-Generator folder not found');
end

fprintf('[Setup] Loading base configuration...\n');
config = load_config('real_signal_dcbias_4k');

% Update GPU setting
config.use_gpu = use_gpu_accel;

%% Parameter validation
if any(beta_sweep < 0 | beta_sweep > 1)
    error('Beta values must be in range [0, 1]');
end

if analysis_freq_hz < 100 || analysis_freq_hz > config.fs/2
    error('Analysis frequency must be in range [100, %.0f] Hz', config.fs/2);
end

n_betas = length(beta_sweep);

%% Initialize result storage
eigenvals_2k = cell(n_betas, 1);
eigenvals_fb = cell(n_betas, 1);
gains_snr = zeros(n_betas, 1);
gains_isr = zeros(n_betas, 1);
cond_nums = zeros(n_betas, 1);
peak_evals = zeros(n_betas, 1);

%% Main loop over beta values
tic;
start_time = datetime('now');

for b_idx = 1:n_betas
    beta_val = beta_sweep(b_idx);
    config.beta = beta_val;

    % Console header
    fprintf('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    fprintf('Processing: β = %.2f  [%d/%d]\n', beta_val, b_idx, n_betas);
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

    % Note: Individual beta outputs saved to temp locations, final comparison figures go to multi_beta_comparison/

    %% Load and process
    if verbose_mode
        fprintf('  → Loading data and generating RIRs...\n');
    end

    [mic_pos, X_target, X_interf, X_noisy, x_target, x_interf, t, Nt, Nmic, r_center] = ...
        load_data_module(config);

    [h_target, h_interf, X_target, X_interf, X_noisy] = ...
        rir_generation_module(config, x_target, x_interf, mic_pos, Nt, Nmic);

    if verbose_mode
        fprintf('  → Running MVDR processing...\n');
    end

    [Yf_tgt, Y_tar_tgt, Y_intnoi_tgt, Yf_int, Y_interf_int, Y_tarnoi_int, ...
     S, S_tar, S_interf, S_intnoi, S_tarnoi, F, T, a_target, a_interf] = ...
        mvdr_processing_module(config, X_noisy, X_target, X_interf, mic_pos, r_center, Nt, Nmic);

    if verbose_mode
        fprintf('  → Computing eigenvalue decompositions...\n');
    end

    %% Eigenvalue extraction at specified analysis frequency
    numFrames = size(Yf_tgt, 2);
    [~, k_analysis] = min(abs(F - analysis_freq_hz));

    t1 = max(1, round(numFrames*0.35));
    t2 = min(numFrames, round(numFrames*0.65));

    Xkf = squeeze(S(k_analysis,:,:)).';
    t2 = min(t2, size(Xkf,2));
    Xloc = Xkf(:, t1:t2);
    Rxx = (Xloc * Xloc') / size(Xloc,2);
    reg = config.epsilon * trace(Rxx) / Nmic;
    Rxx = Rxx + reg * eye(Nmic);

    [~, D] = eig(Rxx);
    evals = sort(diag(D), 'descend');
    eigenvals_2k{b_idx} = evals;
    peak_evals(b_idx) = evals(1);

    %% Full-band eigenvalues
    t1_fb = max(1, round(numFrames*0.35));
    t2_fb = min(size(X_noisy,1), round(Nt*0.65));
    Xloc_fb = X_noisy(t1_fb:t2_fb, :).';
    Rxx_fb = (Xloc_fb * Xloc_fb') / size(Xloc_fb, 2);
    reg_fb = config.epsilon * trace(Rxx_fb) / Nmic;
    Rxx_fb = Rxx_fb + reg_fb * eye(Nmic);

    [~, D_fb] = eig(Rxx_fb);
    evals_fb = sort(diag(D_fb), 'descend');
    eigenvals_fb{b_idx} = evals_fb;
    cond_nums(b_idx) = evals_fb(1) / max(evals_fb(end), eps);

    %% Performance metrics
    win = hamming(config.win_len);

    y_tar_out = real(istft(Y_tar_tgt, config.fs, 'Window', win, ...
        'OverlapLength', config.noverlap, 'FFTLength', config.Nfft));
    y_intnoi_out = real(istft(Y_intnoi_tgt, config.fs, 'Window', win, ...
        'OverlapLength', config.noverlap, 'FFTLength', config.Nfft));

    P_tar = mean(y_tar_out.^2);
    P_intnoi = mean(y_intnoi_out.^2);
    SNR_out = 10*log10(P_tar / (P_intnoi + eps));
    SNR_in = 10*log10(mean(X_target(:,config.ref_mic).^2) / ...
                      (mean(X_interf(:,config.ref_mic).^2) + eps));
    gains_snr(b_idx) = SNR_out - SNR_in;

    y_int_out = real(istft(Y_interf_int, config.fs, 'Window', win, ...
        'OverlapLength', config.noverlap, 'FFTLength', config.Nfft));
    y_tarnoi_out = real(istft(Y_tarnoi_int, config.fs, 'Window', win, ...
        'OverlapLength', config.noverlap, 'FFTLength', config.Nfft));

    ISR_out = 10*log10(mean(y_int_out.^2) / (mean(y_tarnoi_out.^2) + eps));
    ISR_in = 10*log10(mean(X_interf(:,config.ref_mic).^2) / ...
                      (mean(X_target(:,config.ref_mic).^2) + eps));
    gains_isr(b_idx) = ISR_out - ISR_in;

    %% Print metrics
    fprintf('  ✓ SNR Gain:        %.2f dB\n', gains_snr(b_idx));
    fprintf('  ✓ ISR Gain:        %.2f dB\n', gains_isr(b_idx));
    fprintf('  ✓ Condition #:     %.2e\n', cond_nums(b_idx));
    fprintf('  ✓ Peak Eigenvalue: %.2e\n', peak_evals(b_idx));

end

%% =========================================================
%% Generate publication-quality comparison figures
%% =========================================================

fprintf('\n\n╔════════════════════════════════════════════════════════╗\n');
fprintf('║  Generating Publication-Quality Figures                ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

comp_dir = fullfile(config.output_dir, '..', 'multi_beta_comparison');
if ~isfolder(comp_dir)
    mkdir(comp_dir);
end

% Font settings (IEEE)
set(groot, 'defaultAxesFontName','Times New Roman', 'defaultTextFontName','Times New Roman', ...
    'defaultLineLineWidth', 1.3, 'defaultAxesFontSize', 13);

colors = parula(n_betas);
markers_list = {'o', 's', '^', 'd', 'v', 'p', 'h'};

%% Figure 1: Eigenvalue overlay
fprintf('  [1/4] Generating eigenvalue spectrum comparison...\n');
figure('Position', [100, 100, 1000, 700]); clf;

for b_idx = 1:n_betas
    evals = eigenvals_2k{b_idx};
    semilogy(1:length(evals), evals + eps, 'LineWidth', 2.5, 'MarkerSize', 8, ...
        'Color', colors(b_idx,:), 'Marker', markers_list{mod(b_idx-1, 7)+1}, ...
        'DisplayName', sprintf('\\beta = %.2f', beta_sweep(b_idx)), ...
        'MarkerFaceColor', 'w', 'MarkerEdgeColor', colors(b_idx,:));
    hold on;
end

xlabel('Eigenvalue Index', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Eigenvalue Magnitude', 'FontSize', 14, 'FontWeight', 'bold');
title(sprintf('Covariance Eigenspectra at %.0f Hz vs. Reflection Coefficient', ...
    analysis_freq_hz), 'FontSize', 16, 'FontWeight', 'bold');
legend('FontSize', 12, 'Location', 'best', 'Box', 'on', 'Color', 'w', 'EdgeColor', 'k');
grid on; set(gca, 'FontSize', 12, 'LineWidth', 1.5, 'GridAlpha', 0.3);
set(gcf, 'Color', 'white'); set(gca, 'Color', 'white');

save_path_1 = fullfile(comp_dir, sprintf('Eigenspectra_%dHz_comparison.%s', analysis_freq_hz, output_format));
exportgraphics(gcf, save_path_1, 'ContentType','vector', 'BackgroundColor','white', 'Resolution', 600);
fprintf('      → Saved: %s\n', save_path_1);

%% Figure 2: Performance vs beta
fprintf('  [2/4] Generating performance metrics plot...\n');
figure('Position', [100, 100, 1200, 500]); clf;
tiledlayout(1, 3, 'TileSpacing', 'tight', 'Padding', 'tight');

% SNR Gain
nexttile;
plot(beta_sweep, gains_snr, '-o', 'LineWidth', 2.5, 'MarkerSize', 12, 'Color', 'b', 'MarkerFaceColor', [0.7 0.85 1]);
grid on; hold on;
xlabel('Wall Reflection Coefficient (\\beta)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('SNR Gain (dB)', 'FontSize', 12, 'FontWeight', 'bold');
title('Target-Steered: SNR Improvement', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'FontSize', 11, 'LineWidth', 1.5, 'GridAlpha', 0.3);

% ISR Gain
nexttile;
plot(beta_sweep, gains_isr, '-s', 'LineWidth', 2.5, 'MarkerSize', 12, 'Color', 'r', 'MarkerFaceColor', [1 0.7 0.7]);
grid on; hold on;
xlabel('Wall Reflection Coefficient (\\beta)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('ISR Gain (dB)', 'FontSize', 12, 'FontWeight', 'bold');
title('Interference-Steered: ISR Improvement', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'FontSize', 11, 'LineWidth', 1.5, 'GridAlpha', 0.3);

% Condition Number
nexttile;
semilogy(beta_sweep, cond_nums, '-^', 'LineWidth', 2.5, 'MarkerSize', 12, 'Color', [0.5 0 0.5], 'MarkerFaceColor', [0.9 0.8 0.95]);
grid on; hold on;
xlabel('Wall Reflection Coefficient (\\beta)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Condition Number κ', 'FontSize', 12, 'FontWeight', 'bold');
title('Covariance Conditioning', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'FontSize', 11, 'LineWidth', 1.5, 'GridAlpha', 0.3);

set(gcf, 'Color', 'white'); set(findall(gcf,'type','axes'), 'Color', 'white');

save_path_2 = fullfile(comp_dir, sprintf('Performance_metrics_vs_beta.%s', output_format));
exportgraphics(gcf, save_path_2, 'ContentType','vector', 'BackgroundColor','white', 'Resolution', 600);
fprintf('      → Saved: %s\n', save_path_2);

%% Figure 3: Full-band eigenvalues
fprintf('  [3/4] Generating full-band eigenspectra...\n');
figure('Position', [100, 100, 1000, 700]); clf;

for b_idx = 1:n_betas
    evals_fb = eigenvals_fb{b_idx};
    semilogy(1:length(evals_fb), evals_fb + eps, 'LineWidth', 2.5, 'MarkerSize', 8, ...
        'Color', colors(b_idx,:), 'Marker', markers_list{mod(b_idx-1, 7)+1}, ...
        'DisplayName', sprintf('\\beta = %.2f', beta_sweep(b_idx)), ...
        'MarkerFaceColor', 'w', 'MarkerEdgeColor', colors(b_idx,:));
    hold on;
end

xlabel('Eigenvalue Index', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Eigenvalue Magnitude', 'FontSize', 14, 'FontWeight', 'bold');
title('Full-Band Covariance Eigenspectra (Composite Signal)', 'FontSize', 16, 'FontWeight', 'bold');
legend('FontSize', 12, 'Location', 'best', 'Box', 'on', 'Color', 'w', 'EdgeColor', 'k');
grid on; set(gca, 'FontSize', 12, 'LineWidth', 1.5, 'GridAlpha', 0.3);
set(gcf, 'Color', 'white'); set(gca, 'Color', 'white');

save_path_3 = fullfile(comp_dir, sprintf('Eigenspectra_fullband_comparison.%s', output_format));
exportgraphics(gcf, save_path_3, 'ContentType','vector', 'BackgroundColor','white', 'Resolution', 600);
fprintf('      → Saved: %s\n', save_path_3);

%% Figure 4: Summary table as figure
fprintf('  [4/4] Generating summary report...\n');

% Create summary table figure
fig = figure('Position', [100, 100, 1100, 400]); clf;
ax = axes('Parent', fig);
set(ax, 'Visible', 'off');

% Table data
table_data = [
    beta_sweep', ...
    round(gains_snr, 2), ...
    round(gains_isr, 2), ...
    round(cond_nums, 1)
];

% Create table
t = uitable('Parent', fig, 'Data', table_data, ...
    'ColumnName', {'β', 'SNR Gain (dB)', 'ISR Gain (dB)', 'κ(R_xx)'}, ...
    'FontSize', 12, 'FontName', 'Times New Roman', ...
    'ColumnWidth', {100, 120, 120, 120});
t.Position = [50, 50, 900, 300];

% Title
uicontrol('Parent', fig, 'Style', 'text', 'String', ...
    'Multi-Beta MVDR Analysis Summary', ...
    'FontSize', 14, 'FontName', 'Times New Roman', 'FontWeight', 'bold', ...
    'Position', [50, 380, 800, 40]);

set(fig, 'Color', 'white');

save_path_4 = fullfile(comp_dir, sprintf('Analysis_Summary_Table.%s', output_format));
exportgraphics(fig, save_path_4, 'ContentType','vector', 'BackgroundColor','white', 'Resolution', 600);
fprintf('      → Saved: %s\n', save_path_4);

%% Save CSV results
fprintf('  [CSV] Saving numerical results...\n');
results_table = table(beta_sweep(:), gains_snr(:), gains_isr(:), cond_nums(:), peak_evals(:), ...
    'VariableNames', {'Beta', 'SNR_Gain_dB', 'ISR_Gain_dB', 'Condition_Number', 'Peak_Eigenvalue'});

csv_path = fullfile(comp_dir, 'multi_beta_analysis_results.csv');
writetable(results_table, csv_path);
fprintf('      → Saved: %s\n', csv_path);

%% Print summary
elapsed = toc;
fprintf('\n\n╔════════════════════════════════════════════════════════╗\n');
fprintf('║  Analysis Complete!                                    ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

fprintf('Summary:\n');
fprintf('  Total time:        %.1f seconds (%.1f min)\n', elapsed, elapsed/60);
fprintf('  Beta values:       %d\n', n_betas);
fprintf('  Output format:     %s\n', upper(output_format));
fprintf('  Comparison folder: %s\n\n', comp_dir);

fprintf('Results Summary:\n');
fprintf('┌─────────────────────────────────────────────────────────┐\n');
fprintf('│ β    │ SNR Gain (dB) │ ISR Gain (dB) │ κ(R_xx)         │\n');
fprintf('├─────────────────────────────────────────────────────────┤\n');
for b_idx = 1:n_betas
    fprintf('│%.2f │      %.2f      │      %.2f      │  %.2e        │\n', ...
        beta_sweep(b_idx), gains_snr(b_idx), gains_isr(b_idx), cond_nums(b_idx));
end
fprintf('└─────────────────────────────────────────────────────────┘\n\n');

fprintf('Key Observations:\n');
[~, idx_best] = max(gains_snr);
[~, idx_worst] = min(gains_snr);
fprintf('  • Best SNR Gain:   β=%.2f with %.2f dB improvement\n', beta_sweep(idx_best), gains_snr(idx_best));
fprintf('  • Worst SNR Gain:  β=%.2f with %.2f dB improvement\n', beta_sweep(idx_worst), gains_snr(idx_worst));
fprintf('  • SNR Degradation: %.2f dB per 0.1 increase in β\n', ...
    (gains_snr(1) - gains_snr(end)) / (beta_sweep(end) - beta_sweep(1)) * 0.1);

fprintf('\n✓ All figures saved in IEEE publication-quality format (600 DPI)\n');
fprintf('✓ See %s for figures\n\n', comp_dir);

function str = onoff_str(val)
    if val, str = 'ON '; else, str = 'OFF'; end
end
