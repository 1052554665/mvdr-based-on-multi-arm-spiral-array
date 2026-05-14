set(groot, ...
    'defaultAxesFontName','Times New Roman', ...
    'defaultTextFontName','Times New Roman', ...
    'defaultAxesFontSize',20, ...
    'defaultTextFontSize',24, ...
    'defaultLineLineWidth',1.2);

%% MVDR (3D steering: azimuth + elevation) using RIR delays only (no manual delay)
clear; close all; clc;

%% ========== 1. Read mic positions (assume mic_positions.xlsx has X Y columns) ==========
data = readmatrix('mic_positions.xlsx');
X = data(:,1); Y = data(:,2);
Z = zeros(size(X));           % 如果你在Excel里有Z列，把这行改为 Z = data(:,3);
mic_pos = [X Y Z] / 1000;     % 假定Excel单位为 mm，直接换成 m；如果是 cm 用 /100

% center to given array center (optional)
array_center = [2.5 2 1.5];   
mic_pos = mic_pos - mean(mic_pos,1) + array_center;     % 麦克风阵列位置

Nmic = size(mic_pos,1);
r_center = mean(mic_pos,1);   % 阵列中心（用于平面波相位参考）
fprintf('Nmic = %d, mic positions loaded (meters).\n', Nmic);

figure(1);
scatter(X, Y, 40, 'filled');
% scatter3(mic_pos(:,1), mic_pos(:,2), mic_pos(:,3), 40, 'filled'); % 三维阵列图
% axis equal; grid on;
% xlabel('X (m)'); ylabel('Y (m)');
% title('Microphone Array Geometry');

set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'Microphone Array Geometry.pdf', ...
    'ContentType','image', 'BackgroundColor','white');


%% ========== 2. Room, RIR and signals parameters ==========
c = 340;
fs = 16000;
SNR = 30;          % dB
% INR = 5;          % dB (interference relative to target amplitude)
nsample = 4096;    % RIR length

% source 3D positions
s_target = [2.5 1 4];   % used for RIR generation (m)
s_interf  = [2.5 3 4];

% optional GPU acceleration (requires Parallel Computing Toolbox)
use_gpu = true;
gpu_ok = false;
if use_gpu && gpuDeviceCount > 0
    g = gpuDevice;
    gpu_ok = true;
    fprintf('GPU enabled: %s\n', g.Name);
else
    fprintf('GPU not used. Running on CPU.\n');
end

% speed options
use_parallel_rir = true;   % parallelize per-mic RIR generation when possible
mvdr_frame_stride = 2;     % update MVDR weights every N frames (1 = full update)

% source level control for reproducible tone experiments
normalize_source_rms = true;
INR_dB = 0;                 % interference-to-target level before room propagation

% false: one weight per freq (fast), true: per-frame adaptive
if ~exist('mvdr_use_time_varying', 'var')
    mvdr_use_time_varying = false;
end

mvdr_fmax_hz = 8000;             % MVDR upper band; raise to 8 kHz for full-band processing
mvdr_fmin_hz = 300;              % very low frequencies have weak spatial selectivity
mvdr_progress_step = 20;         % print progress every N frequency bins
if ~exist('use_oracle_intnoi_cov', 'var')
    use_oracle_intnoi_cov = true;    % true: use interference+noise covariance (debug upper bound)
end
if ~exist('use_oracle_tarnoi_cov', 'var')
    use_oracle_tarnoi_cov = true;    % true: use target+noise covariance for interference-steered MVDR
end

%% ====================== 可视化房间、声源和麦克风阵列位置 ======================
figure(2); clf;
hold on; grid on; axis equal;

% 1. 绘制房间边界（矩形框）
L = [5 4 6];  % 房间长宽高
xv = [0 L(1) L(1) 0 0];
yv = [0 0 L(2) L(2) 0];
z0 = zeros(size(xv));

% 地面和顶面
plot3(xv, yv, z0, 'k--', 'LineWidth', 1.2);
plot3(xv, yv, L(3)*ones(size(z0)), 'k--', 'LineWidth', 1.2);

% 垂直边
for i = 1:4
    plot3([xv(i) xv(i)], [yv(i) yv(i)], [0 L(3)], 'k--');
end

% 2. 绘制麦克风阵列位置
scatter3(mic_pos(:,1), mic_pos(:,2), mic_pos(:,3), 100, 'filled', 'b');
text(mean(mic_pos(:,1)) + 0.2, mean(mic_pos(:,2)), mean(mic_pos(:,3)) + 0.3, 'Mic Array', ...
     'Color', 'b', 'FontWeight', 'bold');

% 3. 绘制目标声源与干扰源 
scatter3(s_target(1), s_target(2), s_target(3), 60, 'r', 'filled');
text(s_target(1) - 0.6, s_target(2) - 0.6, s_target(3) + 0.4, 'Target', 'Color', 'r', 'FontWeight', 'bold');

scatter3(s_interf(1), s_interf(2), s_interf(3), 60, 'm', 'filled');
text(s_interf(1) + 0.2, s_interf(2), s_interf(3) + 0.1, 'Interference', 'Color', 'm', 'FontWeight', 'bold');

% 4. 可选：连接阵列中心与声源方向
% r_center = mean(mic_pos,1);
plot3([r_center(1) s_target(1)], [r_center(2) s_target(2)], [r_center(3) s_target(3)], 'r--', 'LineWidth', 1.2);
plot3([r_center(1) s_interf(1)], [r_center(2) s_interf(2)], [r_center(3) s_interf(3)], 'm--', 'LineWidth', 1.2);

% 5. 设置视角与标签
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
% title('Room, Sources, and Microphone Array Layout');
view(45, 25);
% legend({'Room boundary','Microphones','Target Source','Interference'}, 'Location','best');
%%
set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'Room, Sources, and Microphone Array Layout.pdf', ...
    'ContentType','image', 'BackgroundColor','white');


%% ========== 3. Fraunhofer check (far-field criterion) ==========
D = 0.15;           % 阵列最大孔径 (m) — 150 mm
f_max = 8000;       % 关心的最高频率（Hz）
lambda_min = c / f_max;
R_far = 2 * D^2 / lambda_min;   % Fraunhofer distance (approx)

fprintf('Array diameter D=%.3f m, f_max=%d Hz, lambda_min=%.3f m\n', D, f_max, lambda_min);
fprintf('Fraunhofer distance R_far ≈ %.3f m\n', R_far);

% compute actual distances of sources to array center
dist_s_target = norm(s_target - r_center);
dist_s_interf  = norm(s_interf  - r_center);

fprintf('Distance target->array center = %.3f m, interference = %.3f m\n', dist_s_target, dist_s_interf);

use_farfield_target = dist_s_target >= R_far;
use_farfield_interf  = dist_s_interf  >= R_far;

if ~use_farfield_target
    fprintf('Warning: target is within near-field (%.3f < %.3f). Using near-field steering for target.\n', dist_s_target, R_far);
else
    fprintf('Target satisfies far-field criterion. Using plane-wave steering for target.\n');
end
if ~use_farfield_interf
    fprintf('Warning: interference is within near-field. Using near-field steering for interference.\n');
else
    fprintf('Interference satisfies far-field criterion. Using plane-wave steering for interference.\n');
end

%% ========== 4. Generate RIRs (per-mic) ==========
% L = [5 4 6];    % room dims
% beta = 0.4;     % wall reflection coefficient
% mtype = 'omnidirectional';  % depends on your rir_generator
% order = -1;
% dim = 3;
% orientation = [pi/2 0];
% hp_filter = 1;

L = [5 4 6];    % room dims
beta = 0;     % wall reflection coefficient
mtype = 'omnidirectional';  % depends on your rir_generator
order = -1;
dim = 3;
orientation = 0;
hp_filter = true;

h_target = zeros(Nmic, nsample);    % 初始化目标和干扰的RIR矩阵
h_interf = zeros(Nmic, nsample);

% beta=0时只有直达声，RIR长度可安全缩短以加速
if beta == 0
    max_dist = max([vecnorm(mic_pos - s_target,2,2); vecnorm(mic_pos - s_interf,2,2)]);
    nsample_direct = max(256, ceil(max_dist / c * fs) + 128);
    nsample_use = min(nsample, nsample_direct);
else
    nsample_use = nsample;
end

if nsample_use < nsample
    h_target = zeros(Nmic, nsample_use);
    h_interf = zeros(Nmic, nsample_use);
    fprintf('Using shortened RIR length: %d (from %d).\n', nsample_use, nsample);
end

fprintf('Generating RIRs for %d microphones (this may take a while)...\n', Nmic);
if use_parallel_rir && license('test', 'Distrib_Computing_Toolbox')
    p = gcp('nocreate');
    if isempty(p)
        try
            parpool('Processes');
        catch ME
            msg = ['Failed to start process-based pool: ' char(ME.message) '. Falling back to serial RIR.'];
            warning('rir:pool', '%s', msg);
            use_parallel_rir = false;
        end
    elseif contains(lower(class(p)), 'thread')
        delete(p);
        try
            parpool('Processes');
        catch ME
            msg = ['Failed to switch to process-based pool: ' char(ME.message) '. Falling back to serial RIR.'];
            warning('rir:pool', '%s', msg);
            use_parallel_rir = false;
        end
    end
end

if use_parallel_rir && license('test', 'Distrib_Computing_Toolbox')
    parfor m = 1:Nmic      % 并行按麦克风生成RIR
        rm = mic_pos(m,:);
        ht = rir_generator(c, fs, rm, s_target, L, beta, nsample_use, mtype, order, dim, orientation, hp_filter);
        hi = rir_generator(c, fs, rm, s_interf, L, beta, nsample_use, mtype, order, dim, orientation, hp_filter);
        if iscolumn(ht), ht = ht.'; end
        if iscolumn(hi), hi = hi.'; end
        h_target(m,:) = ht;
        h_interf(m,:)  = hi;
    end
else
    for m = 1:Nmic      % 循环为每个麦克风生成目标和干扰的RIR，并确保为行向量后存储。
        rm = mic_pos(m,:);
        ht = rir_generator(c, fs, rm, s_target, L, beta, nsample_use, mtype, order, dim, orientation, hp_filter);
        hi = rir_generator(c, fs, rm, s_interf, L, beta, nsample_use, mtype, order, dim, orientation, hp_filter);
        if iscolumn(ht), ht = ht.'; end
        if iscolumn(hi), hi = hi.'; end
        h_target(m,:) = ht;
        h_interf(m,:)  = hi;
    end
end
fprintf('RIR generation done.\n');

%% ========== 5. Generate source signals and convolve with RIR (do NOT add manual delays) ==========
%% ========== Load real source signals (.wav) ==========
[target_sig, fs_t] = audioread('Normal_part92.wav');
[interf_sig, fs_i] = audioread('振安1#反_part46.wav');

% mono conversion
if size(target_sig,2) > 1
    target_sig = mean(target_sig,2);
end
if size(interf_sig,2) > 1
    interf_sig = mean(interf_sig,2);
end

% resample to system fs
if fs_t ~= fs
    target_sig = resample(target_sig, fs, fs_t);
end
if fs_i ~= fs
    interf_sig = resample(interf_sig, fs, fs_i);
end

% length alignment
Nt = min(length(target_sig), length(interf_sig));
x_target = target_sig(1:Nt);
x_interf = interf_sig(1:Nt);

if normalize_source_rms
    x_target = x_target / (sqrt(mean(x_target.^2)) + eps);
    x_interf = x_interf / (sqrt(mean(x_interf.^2)) + eps);
end

% set source INR before room propagation
x_interf = x_interf * 10^(INR_dB/20);

t = (0:Nt-1).' / fs;


Nt = length(x_target);  % 初始化信号矩阵
X_target = zeros(Nt, Nmic);
X_interf  = zeros(Nt, Nmic);

% convolution (same length)
for m = 1:Nmic      % 每个麦克风通道进行卷积，保持相同长度
    X_target(:,m) = conv(x_target, h_target(m,:), 'same');
    X_interf(:,m)  = conv(x_interf,  h_interf(m,:),  'same');
end

% % scale interference by INR (dB) 根据INR缩放干扰信号
% X_interf = X_interf / (10^(INR/20));    

% add noise to meet overall SNR per channel (relative to target)
% 计算目标信号的RMS，根据SNR计算期望的噪声RMS，生成高斯噪声并调整其RMS
% sig_rms = sqrt(mean(X_target.^2, 1));
% desired_noise_rms = sig_rms ./ (10^(SNR/20));
% noise = randn(size(X_target));
% cur_noise_rms = sqrt(mean(noise.^2, 1));
% noise = noise .* (desired_noise_rms ./ cur_noise_rms);

% direct input from loaded target/interference signals (no synthetic noise)
noise = zeros(size(X_target));
X_noisy = X_target + X_interf;
fprintf('Signals prepared from direct target/interference signals (no added noise).\n');

%% ========== SNR BEFORE MVDR (reference: mic 1) ==========
ref_mic = 1;

x_tar_in = X_target(:, ref_mic);
x_intnoi_in = X_interf(:, ref_mic) + noise(:, ref_mic);

P_tar_in = mean(x_tar_in.^2);
P_intnoi_in = mean(x_intnoi_in.^2);
P_int_in = mean(X_interf(:, ref_mic).^2);
P_noise_in = mean(noise(:, ref_mic).^2);

SNR_in_dB = 10*log10(P_tar_in / P_intnoi_in);
SIR_in_dB = 10*log10(P_tar_in / P_int_in);
if P_noise_in > 0
    SNR_noise_in_dB = 10*log10(P_tar_in / P_noise_in);
else
    SNR_noise_in_dB = inf;
end

fprintf('\n===== INPUT SNR =====\n');
fprintf('Input SINR (mic %d, target/(interference+noise)): %.2f dB\n', ref_mic, SNR_in_dB);
fprintf('Input SIR  (target/interference)              : %.2f dB\n', SIR_in_dB);
fprintf('Input SNRn (target/noise only)                : %.2f dB\n', SNR_noise_in_dB);
fprintf('=====================\n');


%% ========== 6. STFT parameters and compute STFT per channel ==========
Nfft = 1024;
win = hamming(512);
noverlap = 256;

% compute STFT channel-wise; we store as S(freq, time, channel)
% 对每个麦克风通道进行STFT，结果存储在S（频率×时间×通道）
for m = 1:Nmic
    [S(:,:,m), F, T] = stft(X_noisy(:,m), fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft, 'FrequencyRange', 'onesided');
end
Fbins = length(F);  % 获取频率点和时间帧
Tframes = length(T);

% limit up to f_limit (mvdr_fmin_hz-mvdr_fmax_hz)
f_start = find(F >= mvdr_fmin_hz, 1, 'first');
f_limit = find(F <= mvdr_fmax_hz, 1, 'last');
if isempty(f_start), f_start = 1; end
if isempty(f_limit), f_limit = Fbins; end
if f_start > f_limit
    f_start = 1;
    f_limit = min(Fbins, find(F <= 5000, 1, 'last'));
end
fprintf('STFT computed: %d freq bins, %d time frames. MVDR bins %d..%d (%.1f-%.1f Hz).\n', ...
    Fbins, Tframes, f_start, f_limit, F(f_start), F(f_limit));


%% ========== STFT of target-only and interference+noise (for SNR eval) ==========
S_tar = zeros(size(S));
S_interf = zeros(size(S));
S_intnoi = zeros(size(S));
S_tarnoi = zeros(size(S));

for m = 1:Nmic
    S_tar(:,:,m) = stft(X_target(:,m), fs, ...
        'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft, 'FrequencyRange', 'onesided');

    S_interf(:,:,m) = stft(X_interf(:,m), fs, ...
        'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft, 'FrequencyRange', 'onesided');

    S_intnoi(:,:,m) = stft(X_interf(:,m) + noise(:,m), fs, ...
        'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft, 'FrequencyRange', 'onesided');

    S_tarnoi(:,:,m) = stft(X_target(:,m) + noise(:,m), fs, ...
        'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft, 'FrequencyRange', 'onesided');
end

if gpu_ok
    S = gpuArray(S);
    S_tar = gpuArray(S_tar);
    S_interf = gpuArray(S_interf);
    S_intnoi = gpuArray(S_intnoi);
    S_tarnoi = gpuArray(S_tarnoi);
end

%% ===== Far-field DOA unit vectors (from source position) =====
u_target = (s_target - r_center).';
u_target = u_target / norm(u_target);

u_interf = (s_interf - r_center).';
u_interf = u_interf / norm(u_interf);


%% ========== 7. Far-field steering vector construction ==========
if gpu_ok
    a_target = gpuArray.zeros(Nmic, Fbins);
    a_interf = gpuArray.zeros(Nmic, Fbins);
else
    a_target = zeros(Nmic, Fbins);
    a_interf = zeros(Nmic, Fbins);
end

for k = 1:Fbins
    freq = F(k);
    if k > f_limit
        continue;
    end

    for m = 1:Nmic
        if use_farfield_target
            phase_t = -2*pi*freq/c * ((mic_pos(m,:) - r_center) * u_target);
        else
            d_m_t = norm(s_target - mic_pos(m,:));
            d_ref_t = norm(s_target - r_center);
            phase_t = -2*pi*freq/c * (d_m_t - d_ref_t);
        end

        if use_farfield_interf
            phase_i = -2*pi*freq/c * ((mic_pos(m,:) - r_center) * u_interf);
        else
            d_m_i = norm(s_interf - mic_pos(m,:));
            d_ref_i = norm(s_interf - r_center);
            phase_i = -2*pi*freq/c * (d_m_i - d_ref_i);
        end

        a_target(m,k) = exp(1j * phase_t);
        a_interf(m,k) = exp(1j * phase_i);
    end

    % normalize (important for MVDR stability)
    a_target(:,k) = a_target(:,k) / norm(a_target(:,k));
    a_interf(:,k) = a_interf(:,k) / norm(a_interf(:,k));
end

if use_farfield_target && use_farfield_interf
    fprintf('Using FAR-FIELD steering vectors for target and interference.\n');
elseif ~use_farfield_target && ~use_farfield_interf
    fprintf('Using NEAR-FIELD steering vectors for target and interference.\n');
else
    fprintf('Using MIXED steering vectors (target/interference far-field flags differ).\n');
end


%% ========== 8. Memory-safe per-frame MVDR (diagonal loading adapted) ==========
fprintf('Running dual MVDR (target-steered + interference-steered) ...\n');

% target-steered branch: keep target, suppress interference+noise
Yf_tgt = squeeze(S(:,:,ref_mic));
Y_tar_tgt = squeeze(S_tar(:,:,ref_mic));
Y_intnoi_tgt = squeeze(S_intnoi(:,:,ref_mic));

% interference-steered branch: keep interference, suppress target+noise
Yf_int = squeeze(S(:,:,ref_mic));
Y_interf_int = squeeze(S_interf(:,:,ref_mic));
Y_tarnoi_int = squeeze(S_tarnoi(:,:,ref_mic));

% Tunable parameters (you can try Mavg=21/epsilon=1e-2, or smaller)
Mavg = 31;                     % averaging window for covariance (frames)
epsilon = 1e-4;                % base loading (relative scale)

% optional shrinkage factor for additional stability (set 0 to disable)
shrink_alpha = 0.01;           % 0..0.3 typical; 0 means no shrinkage

tic
%% 对每个频率和时间帧，用局部滑窗估计协方差矩阵（目标指向分支）
for k = f_start:f_limit
    Xkf = squeeze(S(k,:,:)).';   % Nmic x Tframes
    Xkf_tar = squeeze(S_tar(k,:,:)).';        % Nmic x Tframes
    Xkf_intnoi = squeeze(S_intnoi(k,:,:)).';  % Nmic x Tframes
    if use_oracle_intnoi_cov
        Xkf_cov = Xkf_intnoi;
    else
        Xkf_cov = Xkf;
    end
    at = a_target(:,k);
    if norm(at) < 1e-12
        continue;
    end

    if mvdr_use_time_varying
        w_last = zeros(Nmic,1, 'like', at);
        for n = 1:Tframes
            % 按步长更新权重，其余帧复用上一次权重以降低求解次数
            if n == 1 || mod(n-1, mvdr_frame_stride) == 0
                t1 = max(1, n - floor(Mavg/2));
                t2 = min(Tframes, n + floor(Mavg/2));
                Xloc = Xkf_cov(:, t1:t2);

                Rxx = (Xloc * Xloc') / size(Xloc,2);

                if shrink_alpha > 0
                    mu = trace(Rxx) / Nmic;
                    Rxx = (1 - shrink_alpha) * Rxx + shrink_alpha * mu * eye(Nmic, 'like', Rxx);
                end

                Rxx = Rxx + epsilon * trace(Rxx)/Nmic * eye(Nmic, 'like', Rxx);
                w = Rxx \ at;
                denom = at' * w;
                if abs(denom) < 1e-12
                    w = zeros(Nmic,1, 'like', at);
                else
                    w = w / denom;
                end
                w_last = w;
            end

            Yf_tgt(k,n) = w_last' * Xkf(:,n);
            Y_tar_tgt(k,n) = w_last' * Xkf_tar(:,n);
            Y_intnoi_tgt(k,n) = w_last' * Xkf_intnoi(:,n);
        end
    else
        % fast mode: one covariance/weight per frequency bin
        Rxx = (Xkf_cov * Xkf_cov') / Tframes;
        if shrink_alpha > 0
            mu = trace(Rxx) / Nmic;
            Rxx = (1 - shrink_alpha) * Rxx + shrink_alpha * mu * eye(Nmic, 'like', Rxx);
        end
        Rxx = Rxx + epsilon * trace(Rxx)/Nmic * eye(Nmic, 'like', Rxx);
        w = Rxx \ at;
        denom = at' * w;
        if abs(denom) < 1e-12
            w = zeros(Nmic,1, 'like', at);
        else
            w = w / denom;
        end

        Yf_tgt(k,:) = w' * Xkf;
        Y_tar_tgt(k,:) = w' * Xkf_tar;
        Y_intnoi_tgt(k,:) = w' * Xkf_intnoi;
    end

    if mod(k, mvdr_progress_step) == 0 || k == f_limit
        fprintf('MVDR target-beam progress: %d/%d bins (%.1f%%), elapsed %.1fs\n', k, f_limit, 100*k/f_limit, toc);
    end
end

%% 对每个频率和时间帧，用局部滑窗估计协方差矩阵（干扰指向分支）
for k = f_start:f_limit
    Xkf = squeeze(S(k,:,:)).';   % Nmic x Tframes
    Xkf_interf = squeeze(S_interf(k,:,:)).';      % Nmic x Tframes
    Xkf_tarnoi = squeeze(S_tarnoi(k,:,:)).';      % Nmic x Tframes
    if use_oracle_tarnoi_cov
        Xkf_cov = Xkf_tarnoi;
    else
        Xkf_cov = Xkf;
    end
    ai = a_interf(:,k);
    if norm(ai) < 1e-12
        continue;
    end

    if mvdr_use_time_varying
        w_last = zeros(Nmic,1, 'like', ai);
        for n = 1:Tframes
            if n == 1 || mod(n-1, mvdr_frame_stride) == 0
                t1 = max(1, n - floor(Mavg/2));
                t2 = min(Tframes, n + floor(Mavg/2));
                Xloc = Xkf_cov(:, t1:t2);

                Rxx = (Xloc * Xloc') / size(Xloc,2);

                if shrink_alpha > 0
                    mu = trace(Rxx) / Nmic;
                    Rxx = (1 - shrink_alpha) * Rxx + shrink_alpha * mu * eye(Nmic, 'like', Rxx);
                end

                Rxx = Rxx + epsilon * trace(Rxx)/Nmic * eye(Nmic, 'like', Rxx);
                w = Rxx \ ai;
                denom = ai' * w;
                if abs(denom) < 1e-12
                    w = zeros(Nmic,1, 'like', ai);
                else
                    w = w / denom;
                end
                w_last = w;
            end

            Yf_int(k,n) = w_last' * Xkf(:,n);
            Y_interf_int(k,n) = w_last' * Xkf_interf(:,n);
            Y_tarnoi_int(k,n) = w_last' * Xkf_tarnoi(:,n);
        end
    else
        Rxx = (Xkf_cov * Xkf_cov') / Tframes;
        if shrink_alpha > 0
            mu = trace(Rxx) / Nmic;
            Rxx = (1 - shrink_alpha) * Rxx + shrink_alpha * mu * eye(Nmic, 'like', Rxx);
        end
        Rxx = Rxx + epsilon * trace(Rxx)/Nmic * eye(Nmic, 'like', Rxx);
        w = Rxx \ ai;
        denom = ai' * w;
        if abs(denom) < 1e-12
            w = zeros(Nmic,1, 'like', ai);
        else
            w = w / denom;
        end

        Yf_int(k,:) = w' * Xkf;
        Y_interf_int(k,:) = w' * Xkf_interf;
        Y_tarnoi_int(k,:) = w' * Xkf_tarnoi;
    end

    if mod(k, mvdr_progress_step) == 0 || k == f_limit
        fprintf('MVDR interf-beam progress: %d/%d bins (%.1f%%), elapsed %.1fs\n', k, f_limit, 100*k/f_limit, toc);
    end
end
toc

if gpu_ok
    % gather once before diagnostics/ISTFT to avoid repeated host-device sync
    Yf_tgt = gather(Yf_tgt);
    Y_tar_tgt = gather(Y_tar_tgt);
    Y_intnoi_tgt = gather(Y_intnoi_tgt);
    Yf_int = gather(Yf_int);
    Y_interf_int = gather(Y_interf_int);
    Y_tarnoi_int = gather(Y_tarnoi_int);
    S = gather(S);
    S_interf = gather(S_interf);
    S_intnoi = gather(S_intnoi);
    S_tarnoi = gather(S_tarnoi);
    a_target = gather(a_target);
    a_interf = gather(a_interf);
end

% legacy aliases (keep downstream diagnostics compatible with original script)
Yf = Yf_tgt;
Y_tar = Y_tar_tgt;
Y_intnoi = Y_intnoi_tgt;


%% ========== Diagnostic for 2 kHz and Rxx (place right after MVDR loop, before ISTFT) ==========
% find freq bin for 2kHz and 1kHz
[~, k2] = min(abs(F - 2000));

% pick a robust frame interval (middle 30% of frames) for Rxx estimate
numFrames = size(Yf,2);
t1_diag = max(1, round(numFrames*0.35));
t2_diag = min(numFrames, round(numFrames*0.65));
if t1_diag >= t2_diag
    t1_diag = 1; t2_diag = numFrames;
end

% get Xkf for k2 (freq x time x channel earlier)
Xkf_k2 = squeeze(S(k2,:,:)).';   % Nmic x Tframes
% safe bounds
t2_diag = min(t2_diag, size(Xkf_k2,2));
Xloc_diag = Xkf_k2(:, t1_diag:t2_diag);
Rxx_diag = (Xloc_diag * Xloc_diag') / size(Xloc_diag,2);
reg_diag = epsilon * trace(Rxx_diag) / Nmic;
Rxx_diag = Rxx_diag + reg_diag * eye(Nmic, 'like', Rxx_diag);

% eigenspectrum
[~, D] = eig(Rxx_diag);
evals = sort(diag(D), 'descend');
figure(3); plot(1:Nmic, 10*log10(evals + eps), '-o');
xlabel('Index'); ylabel('Eigenvalue (dB)'); 

set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'Rxx eig-spectrum at 2 kHz.pdf', 'Resolution',300,...
    'ContentType','image', 'BackgroundColor','white');
% title('Rxx eig-spectrum at ~2 kHz');

% compute MVDR weight used at k2 for the center frame
center_frame = round(numFrames/2);
t1c = max(1, center_frame - floor(Mavg/2));
t2c = min(numFrames, center_frame + floor(Mavg/2));
Xlocc = Xkf_k2(:, t1c:t2c);
Rxxc = (Xlocc * Xlocc') / size(Xlocc,2);
if shrink_alpha > 0
    mu_c = trace(Rxxc) / Nmic;
    Rxxc = (1 - shrink_alpha) * Rxxc + shrink_alpha * mu_c * eye(Nmic, 'like', Rxxc);
end
Rxxc = Rxxc + (epsilon * trace(Rxxc) / Nmic) * eye(Nmic, 'like', Rxxc);

at_k2 = a_target(:, k2);
w_diag = Rxxc \ at_k2;
denom_diag = (at_k2' * w_diag);
if abs(denom_diag) < 1e-12
    Wdiag = zeros(size(w_diag));
else
    Wdiag = w_diag ./ denom_diag;
end

% beampattern at 2 kHz over azimuth (elev=0)
azs = -180:1:180;
resp = zeros(size(azs));
for ii = 1:length(azs)
    d_try = [cosd(0)*cosd(azs(ii)); cosd(0)*sind(azs(ii)); sind(0)];
    a_try = exp(-1j*2*pi*F(k2) * ((mic_pos - r_center) * d_try) / c);
    a_try = a_try / norm(a_try);
    resp(ii) = 20*log10(abs(Wdiag' * a_try) + eps);
end
figure(4); plot(azs, resp); grid on;
xlabel('Azimuth (deg)'); ylabel('Response (dB)'); 
% title(sprintf('Beampattern at %.1f Hz', F(k2)));

set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'Beampattern at 2k Hz.pdf', 'Resolution',300,...
    'ContentType','image', 'BackgroundColor','white');
fprintf('Diagnostic done. See eig-spectrum and beampattern for 2 kHz.\n');



%% ========== Additional Diagnostic for 4 kHz (interference frequency) ==========
% find freq bin for 4kHz
[~, k4] = min(abs(F - 4000));
center_frame = round(numFrames/2);
t1c_4k = max(1, center_frame - floor(Mavg/2));
t2c_4k = min(numFrames, center_frame + floor(Mavg/2));

% Use interference+noise covariance if oracle mode is enabled (consistent with MVDR loop)
if use_oracle_intnoi_cov
    Xkf_k4 = squeeze(S_intnoi(k4,:,:)).';   % Nmic x Tframes (interference + noise only)
else
    Xkf_k4 = squeeze(S(k4,:,:)).';          % Nmic x Tframes (total signal)
end

t2c_4k = min(t2c_4k, size(Xkf_k4,2));
Xlocc_4k = Xkf_k4(:, t1c_4k:t2c_4k);
Rxxc_4k = (Xlocc_4k * Xlocc_4k') / size(Xlocc_4k,2);

if shrink_alpha > 0
    mu_c_4k = trace(Rxxc_4k) / Nmic;
    Rxxc_4k = (1 - shrink_alpha) * Rxxc_4k + shrink_alpha * mu_c_4k * eye(Nmic, 'like', Rxxc_4k);
end

% Try smaller diagonal loading for better suppression
epsilon_4k = epsilon * 0.1;  % Reduce loading by 10x for diagnostic
Rxxc_4k = Rxxc_4k + (epsilon_4k * trace(Rxxc_4k) / Nmic) * eye(Nmic, 'like', Rxxc_4k);

at_k4 = a_target(:, k4);

% Check condition number to diagnose ill-conditioning
cond_num = cond(Rxxc_4k);
fprintf('\n===== 4 kHz Covariance Diagnostics =====\n');
fprintf('Condition number: %.2e\n', cond_num);
if use_oracle_intnoi_cov
    fprintf('Using interference+noise covariance\n');
else
    fprintf('Using total signal covariance\n');
end
fprintf('Diagonal loading: %.2e\n', epsilon_4k * trace(Rxxc_4k) / Nmic);

w_diag_4k = Rxxc_4k \ at_k4;
denom_diag_4k = (at_k4' * w_diag_4k);
if abs(denom_diag_4k) < 1e-12
    Wdiag_4k = zeros(size(w_diag_4k));
    fprintf('Warning: MVDR weight computation failed (denominator too small)\n');
else
    Wdiag_4k = w_diag_4k ./ denom_diag_4k;
end

% Check beamformer response in target and interference directions
resp_target_check = 20*log10(abs(Wdiag_4k' * at_k4) + eps);
ai_k4 = a_interf(:, k4);
resp_interf_check = 20*log10(abs(Wdiag_4k' * ai_k4) + eps);
fprintf('Beam response at target DOA:  %.2f dB\n', resp_target_check);
fprintf('Beam response at interferer DOA: %.2f dB\n', resp_interf_check);
fprintf('Suppression (target - interf): %.2f dB\n', resp_target_check - resp_interf_check);
fprintf('========================================\n');

% beampattern at 4 kHz over azimuth
azs_4k = -180:1:180;
resp_4k = zeros(size(azs_4k));
for ii = 1:length(azs_4k)
    d_try = [cosd(0)*cosd(azs_4k(ii)); cosd(0)*sind(azs_4k(ii)); sind(0)];
    a_try = exp(-1j*2*pi*F(k4) * ((mic_pos - r_center) * d_try) / c);
    a_try = a_try / norm(a_try);
    resp_4k(ii) = 20*log10(abs(Wdiag_4k' * a_try) + eps);
end

figure(5); plot(azs_4k, resp_4k); grid on;
xlabel('Azimuth (deg)'); ylabel('Response (dB)'); 
title(sprintf('Beampattern at %.1f Hz (Interference)', F(k4)));

% Mark target and interference directions
[~, az_target] = min(abs(azs_4k - atan2d(u_target(2), u_target(1))));
[~, az_interf] = min(abs(azs_4k - atan2d(u_interf(2), u_interf(1))));
hold on;
plot(azs_4k(az_target), resp_4k(az_target), 'r^', 'MarkerSize', 10, 'LineWidth', 2);
plot(azs_4k(az_interf), resp_4k(az_interf), 'mv', 'MarkerSize', 10, 'LineWidth', 2);
legend({'Beam response', 'Target direction', 'Interference direction'});

set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'Beampattern at 4k Hz.pdf', 'Resolution',300,...
    'ContentType','image', 'BackgroundColor','white');

fprintf('4 kHz diagnostic complete. Check beampattern to verify interference suppression.\n');

%% ========== 9. ISTFT and normalization ==========
y_mvdr_target = real(istft(Yf_tgt, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft));
y_mvdr_interf = real(istft(Yf_int, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft));

% robust length alignment   调整输出长度与输入相同
if length(y_mvdr_target) >= Nt
    y_mvdr_target = y_mvdr_target(1:Nt);
else
    y_mvdr_target = [y_mvdr_target; zeros(Nt - length(y_mvdr_target), 1)];
end
if length(y_mvdr_interf) >= Nt
    y_mvdr_interf = y_mvdr_interf(1:Nt);
else
    y_mvdr_interf = [y_mvdr_interf; zeros(Nt - length(y_mvdr_interf), 1)];
end

% energy normalization (avoid clipping)
y_mvdr_target = y_mvdr_target / max(abs(y_mvdr_target) + 1e-12);
y_mvdr_interf = y_mvdr_interf / max(abs(y_mvdr_interf) + 1e-12);

% legacy alias
y_mvdr = y_mvdr_target;


%% ========== ISTFT for metric evaluation ==========
y_tar_out_target = real(istft(Y_tar_tgt, fs, ...
    'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft));
y_intnoi_out_target = real(istft(Y_intnoi_tgt, fs, ...
    'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft));

y_interf_out_interf = real(istft(Y_interf_int, fs, ...
    'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft));
y_tarnoi_out_interf = real(istft(Y_tarnoi_int, fs, ...
    'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft));

%% ===== Length alignment (SAFE) =====
Lsig = min([length(y_mvdr_target), length(y_mvdr_interf), ...
    length(y_tar_out_target), length(y_intnoi_out_target), ...
    length(y_interf_out_interf), length(y_tarnoi_out_interf)]);

y_mvdr_target = y_mvdr_target(1:Lsig);
y_mvdr_interf = y_mvdr_interf(1:Lsig);
y_mvdr = y_mvdr(1:Lsig);

y_tar_out_target = y_tar_out_target(1:Lsig);
y_intnoi_out_target = y_intnoi_out_target(1:Lsig);
y_interf_out_interf = y_interf_out_interf(1:Lsig);
y_tarnoi_out_interf = y_tarnoi_out_interf(1:Lsig);


%% ========== Target-steered MVDR performance ==========
P_tar_out = mean(y_tar_out_target.^2);
P_intnoi_out = mean(y_intnoi_out_target.^2);

SNR_out_dB = 10*log10(P_tar_out / P_intnoi_out);
SNR_gain_dB = SNR_out_dB - SNR_in_dB;

fprintf('\n===== TARGET-STEERED MVDR =====\n');
fprintf('Input  SNR (mic %d, target/(interf+noise)): %.2f dB\n', ref_mic, SNR_in_dB);
fprintf('Output SNR (target beam)                  : %.2f dB\n', SNR_out_dB);
fprintf('SNR Improvement                           : %.2f dB\n', SNR_gain_dB);
fprintf('================================\n');


%% ========== Interference-steered MVDR performance ==========
P_interf_in = mean(X_interf(:, ref_mic).^2);
P_tarnoi_in = mean((X_target(:, ref_mic) + noise(:, ref_mic)).^2);
ISR_in_dB = 10*log10(P_interf_in / P_tarnoi_in);

P_interf_out = mean(y_interf_out_interf.^2);
P_tarnoi_out = mean(y_tarnoi_out_interf.^2);
ISR_out_dB = 10*log10(P_interf_out / P_tarnoi_out);
ISR_gain_dB = ISR_out_dB - ISR_in_dB;

fprintf('\n===== INTERFERENCE-STEERED MVDR =====\n');
fprintf('Input  ISR (mic %d, interf/(target+noise)): %.2f dB\n', ref_mic, ISR_in_dB);
fprintf('Output ISR (interference beam)           : %.2f dB\n', ISR_out_dB);
fprintf('ISR Improvement                          : %.2f dB\n', ISR_gain_dB);
fprintf('========================================\n');


%% ========== Improvement visualization ==========
figure(6);
bar([SNR_in_dB, SNR_out_dB; ISR_in_dB, ISR_out_dB]);
set(gca,'XTickLabel',{'Target beam','Interference beam'});
ylabel('Ratio (dB)');
legend({'Before MVDR','After MVDR'}, 'Location', 'best');
grid on;

set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'MVDR Ratio Improvement (Dual Beam).pdf', 'Resolution',300,...
    'ContentType','image', 'BackgroundColor','white');


% ========== 10. Diagnostics: waveforms, spectrograms, PSD ==========
figure(7);
t1 = (0:length(X_target(:,1))-1)/fs;
plot(t1, X_target(:,1));
xlabel('Time (s)');
ylabel('Amplitude');
set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'Target signal.pdf', ...
    'ContentType','image', 'BackgroundColor','white');

figure(8);
t2 = (0:length(X_noisy(:,1))-1)/fs;
plot(t2, X_noisy(:,1));
xlabel('Time (s)');
ylabel('Amplitude');
set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'Noisy signal.pdf', ...
    'ContentType','image', 'BackgroundColor','white');

figure(9);
t3 = (0:length(y_mvdr_target)-1)/fs;
plot(t3, y_mvdr_target, 'b'); hold on;
plot(t3, y_mvdr_interf, 'm');
xlabel('Time (s)');
ylabel('Amplitude');
legend({'Target-steered output','Interference-steered output'});
set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'MVDR dual outputs.pdf', ...
    'ContentType','image', 'BackgroundColor','white');


figure(10);
spectrogram(X_target(:,1), hamming(256), 128, 512, fs, 'yaxis');
xlabel('Time (ms)');
ylabel('Frequency(kHz)');
c = colorbar;
c.Label.String = 'Power/Frequency (dB/Hz)';
set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'Target spectrogram.pdf', 'Resolution',300,...
    'ContentType','image', 'BackgroundColor','white');

figure(11);
spectrogram(X_noisy(:,1), hamming(256), 128, 512, fs, 'yaxis');
xlabel('Time (ms)');
ylabel('Frequency(kHz)');
c = colorbar;
c.Label.String = 'Power/Frequency (dB/Hz)';
set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'Noisy spectrogram.pdf', 'Resolution',300, ...
    'ContentType','image', 'BackgroundColor','white');

figure(12);
spectrogram(y_mvdr_target, hamming(256), 128, 512, fs, 'yaxis');
xlabel('Time (ms)');
ylabel('Frequency(kHz)');
c = colorbar;
c.Label.String = 'Power/Frequency (dB/Hz)';
set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'MVDR target-beam spectrogram.pdf', 'Resolution',300,...
    'ContentType','image', 'BackgroundColor','white');

figure(13);
spectrogram(y_mvdr_interf, hamming(256), 128, 512, fs, 'yaxis');
xlabel('Time (ms)');
ylabel('Frequency(kHz)');
c = colorbar;
c.Label.String = 'Power/Frequency (dB/Hz)';
set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'MVDR interference-beam spectrogram.pdf', 'Resolution',300,...
    'ContentType','image', 'BackgroundColor','white');

% 处理前后频谱图（目标指向）
figure(14); clf;
subplot(1,2,1);
spectrogram(X_noisy(:,1), hamming(256), 128, 512, fs, 'yaxis');
title('Before MVDR (mic1)');
xlabel('Time (ms)'); ylabel('Frequency(kHz)');
subplot(1,2,2);
spectrogram(y_mvdr_target, hamming(256), 128, 512, fs, 'yaxis');
title('After MVDR (Target-steered)');
xlabel('Time (ms)'); ylabel('Frequency(kHz)');
ax17 = findall(gcf, 'Type', 'Axes');
clim17 = max(cell2mat(arrayfun(@(ax) ax.CLim, ax17, 'UniformOutput', false)), [], 1);
arrayfun(@(ax) caxis(ax, clim17), ax17);
set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'Before-After Spectrogram (Target Beam).pdf', 'Resolution',300,...
    'ContentType','image', 'BackgroundColor','white');

% 处理前后频谱图（干扰指向）
figure(15); clf;
subplot(1,2,1);
spectrogram(X_noisy(:,1), hamming(256), 128, 512, fs, 'yaxis');
title('Before MVDR (mic1)');
xlabel('Time (ms)'); ylabel('Frequency(kHz)');
subplot(1,2,2);
spectrogram(y_mvdr_interf, hamming(256), 128, 512, fs, 'yaxis');
title('After MVDR (Interference-steered)');
xlabel('Time (ms)'); ylabel('Frequency(kHz)');
ax18 = findall(gcf, 'Type', 'Axes');
clim18 = max(cell2mat(arrayfun(@(ax) ax.CLim, ax18, 'UniformOutput', false)), [], 1);
arrayfun(@(ax) caxis(ax, clim18), ax18);
set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'Before-After Spectrogram (Interference Beam).pdf', 'Resolution',300,...
    'ContentType','image', 'BackgroundColor','white');


% PSD comparison
nfft_psd = 4096;
[Pxx_in, Fp] = pwelch(X_noisy(:,1), hann(1024), 512, nfft_psd, fs);
[Pxx_out_target, ~] = pwelch(y_mvdr_target, hann(1024), 512, nfft_psd, fs);
[Pxx_out_interf, ~] = pwelch(y_mvdr_interf, hann(1024), 512, nfft_psd, fs);

figure(16);
plot(Fp, 10*log10(Pxx_in + eps), 'k'); hold on;
plot(Fp, 10*log10(Pxx_out_target + eps), 'b', 'LineWidth', 1.5);
plot(Fp, 10*log10(Pxx_out_interf + eps), 'm', 'LineWidth', 1.5);
xlim([0 fs/2]);
xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
legend({'Before MVDR','After Target-steered','After Interference-steered'});
grid on;
set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'PSD Fullband Dual MVDR.pdf', 'Resolution',300,...
    'ContentType','image', 'BackgroundColor','white');

figure(17); clf;
plot(Fp, 10*log10(Pxx_in + eps), 'k'); hold on;
plot(Fp, 10*log10(Pxx_out_target + eps), 'b', 'LineWidth', 1.5);
xlim([1800 2200]);
xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
legend({'Before MVDR','After Target-steered'});
set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'PSD around 2 kHz (Target Beam).pdf', 'Resolution',300,...
    'ContentType','image', 'BackgroundColor','white');

figure(18); clf;
plot(Fp, 10*log10(Pxx_in + eps), 'k'); hold on;
plot(Fp, 10*log10(Pxx_out_target + eps), 'b', 'LineWidth', 1.5);
plot(Fp, 10*log10(Pxx_out_interf + eps), 'm', 'LineWidth', 1.5);
xlim([3500 4500]);
xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
legend({'Before MVDR','After Target-steered','After Interference-steered'});
title('PSD around 4 kHz');
grid on;

[~, idx_4k] = min(abs(Fp - 4000));
psd_in_4k = 10*log10(Pxx_in(idx_4k) + eps);
psd_tgt_4k = 10*log10(Pxx_out_target(idx_4k) + eps);
psd_int_4k = 10*log10(Pxx_out_interf(idx_4k) + eps);

fprintf('\n===== 4 kHz PSD CHECK =====\n');
fprintf('Before MVDR (mic1): %.2f dB/Hz\n', psd_in_4k);
fprintf('After Target-steered: %.2f dB/Hz\n', psd_tgt_4k);
fprintf('After Interference-steered: %.2f dB/Hz\n', psd_int_4k);
fprintf('============================\n');

set(gca, 'LineWidth', 1);
set(gcf,'InvertHardcopy','off'); set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, 'PSD around 4 kHz (Dual Beam).pdf', 'Resolution',300,...
    'ContentType','image', 'BackgroundColor','white');