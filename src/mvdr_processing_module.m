%% ============== MVDR Processing Module ==============
% Core MVDR beamforming with target-steered and interference-steered branches

function [Yf_tgt, Y_tar_tgt, Y_intnoi_tgt, Yf_int, Y_interf_int, Y_tarnoi_int, ...
          S, S_tar, S_interf, S_intnoi, S_tarnoi, F, T, a_target, a_interf] = ...
          mvdr_processing_module(config, X_noisy, X_target, X_interf, mic_pos, r_center, Nt, Nmic)

fprintf('\n[MVDR] Starting MVDR processing...\n');

%% 1. Compute STFT for all signals
fprintf('[MVDR] Computing STFT...\n');

win = hamming(config.win_len);

% Compute first STFT to determine dimensions
[S_temp, F, T] = stft(X_noisy(:,1), config.fs, ...
    'Window', win, 'OverlapLength', config.noverlap, ...
    'FFTLength', config.Nfft, 'FrequencyRange', 'onesided');

Fbins = length(F);
Tframes = length(T);

% Pre-allocate full S array with correct dimensions
S = zeros(Fbins, Tframes, Nmic);
S(:,:,1) = S_temp;

% Compute STFT for remaining channels
for m = 2:Nmic
    S(:,:,m) = stft(X_noisy(:,m), config.fs, ...
        'Window', win, 'OverlapLength', config.noverlap, ...
        'FFTLength', config.Nfft, 'FrequencyRange', 'onesided');
end

fprintf('[MVDR] STFT: %d freq bins, %d time frames\n', Fbins, Tframes);

%% Compute STFT for clean components
fprintf('[MVDR] Computing STFTs for metric computation...\n');

S_tar = zeros(size(S));
S_interf = zeros(size(S));
S_intnoi = zeros(size(S));
S_tarnoi = zeros(size(S));
noise = zeros(size(X_noisy));  % No synthetic noise

for m = 1:Nmic
    S_tar(:,:,m) = stft(X_target(:,m), config.fs, ...
        'Window', win, 'OverlapLength', config.noverlap, 'FFTLength', config.Nfft, ...
        'FrequencyRange', 'onesided');
    S_interf(:,:,m) = stft(X_interf(:,m), config.fs, ...
        'Window', win, 'OverlapLength', config.noverlap, 'FFTLength', config.Nfft, ...
        'FrequencyRange', 'onesided');
    S_intnoi(:,:,m) = stft(X_interf(:,m) + noise(:,m), config.fs, ...
        'Window', win, 'OverlapLength', config.noverlap, 'FFTLength', config.Nfft, ...
        'FrequencyRange', 'onesided');
    S_tarnoi(:,:,m) = stft(X_target(:,m) + noise(:,m), config.fs, ...
        'Window', win, 'OverlapLength', config.noverlap, 'FFTLength', config.Nfft, ...
        'FrequencyRange', 'onesided');
end

%% Move to GPU if available
gpu_ok = config.use_gpu && gpuDeviceCount > 0;
if gpu_ok
    g = gpuDevice;
    fprintf('[MVDR] Using GPU: %s\n', g.Name);
    S = gpuArray(S);
    S_tar = gpuArray(S_tar);
    S_interf = gpuArray(S_interf);
    S_intnoi = gpuArray(S_intnoi);
    S_tarnoi = gpuArray(S_tarnoi);
else
    fprintf('[MVDR] GPU not used, running on CPU\n');
end

%% 2. Determine frequency band for MVDR
f_start = find(F >= config.mvdr_fmin_hz, 1, 'first');
f_limit = find(F <= config.mvdr_fmax_hz, 1, 'last');
if isempty(f_start), f_start = 1; end
if isempty(f_limit), f_limit = Fbins; end
if f_start > f_limit
    f_start = 1;
    f_limit = min(Fbins, find(F <= 5000, 1, 'last'));
end
fprintf('[MVDR] Processing frequency range: %.1f-%.1f Hz (bins %d-%d)\n', ...
    F(f_start), F(f_limit), f_start, f_limit);

%% 3. Compute far-field steering vectors
fprintf('[MVDR] Computing steering vectors...\n');

u_target = (config.s_target - r_center).';
u_target = u_target / norm(u_target);

u_interf = (config.s_interf - r_center).';
u_interf = u_interf / norm(u_interf);

% Check far-field criterion
f_max = config.mvdr_fmax_hz;
lambda_min = config.c / f_max;
R_far = 2 * config.array_diameter^2 / lambda_min;

dist_target = norm(config.s_target - r_center);
dist_interf = norm(config.s_interf - r_center);

use_farfield_target = dist_target >= R_far;
use_farfield_interf = dist_interf >= R_far;

fprintf('[MVDR] Target: %s-field | Interference: %s-field\n', ...
    iif(use_farfield_target, 'Far', 'Near'), ...
    iif(use_farfield_interf, 'Far', 'Near'));

if gpu_ok
    a_target = gpuArray.zeros(Nmic, Fbins);
    a_interf = gpuArray.zeros(Nmic, Fbins);
else
    a_target = zeros(Nmic, Fbins);
    a_interf = zeros(Nmic, Fbins);
end

for k = 1:Fbins
    if k > f_limit
        continue;
    end
    
    freq = F(k);
    for m = 1:Nmic
        if use_farfield_target
            phase_t = -2*pi*freq/config.c * ((mic_pos(m,:) - r_center) * u_target);
        else
            d_m_t = norm(config.s_target - mic_pos(m,:));
            d_ref_t = norm(config.s_target - r_center);
            phase_t = -2*pi*freq/config.c * (d_m_t - d_ref_t);
        end
        
        if use_farfield_interf
            phase_i = -2*pi*freq/config.c * ((mic_pos(m,:) - r_center) * u_interf);
        else
            d_m_i = norm(config.s_interf - mic_pos(m,:));
            d_ref_i = norm(config.s_interf - r_center);
            phase_i = -2*pi*freq/config.c * (d_m_i - d_ref_i);
        end
        
        a_target(m,k) = exp(1j * phase_t);
        a_interf(m,k) = exp(1j * phase_i);
    end
    
    % Normalize
    a_target(:,k) = a_target(:,k) / norm(a_target(:,k));
    a_interf(:,k) = a_interf(:,k) / norm(a_interf(:,k));
end

fprintf('[MVDR] Steering vectors computed.\n');

%% 4. Run MVDR beamforming
fprintf('[MVDR] Running dual-branch MVDR beamforming...\n');
fprintf('[MVDR] (This may take a few minutes...)\n');

tic;
[Yf_tgt, Y_tar_tgt, Y_intnoi_tgt] = mvdr_branch(S, S_tar, S_intnoi, a_target, ...
    config, Fbins, Tframes, f_start, f_limit, Nmic, 'target-steered');

[Yf_int, Y_interf_int, Y_tarnoi_int] = mvdr_branch(S, S_interf, S_tarnoi, a_interf, ...
    config, Fbins, Tframes, f_start, f_limit, Nmic, 'interference-steered');
elapsed = toc;

fprintf('[MVDR] Beamforming completed in %.2f seconds\n', elapsed);

%% 5. Gather from GPU if needed
if gpu_ok
    fprintf('[MVDR] Gathering results from GPU...\n');
    Yf_tgt = gather(Yf_tgt);
    Y_tar_tgt = gather(Y_tar_tgt);
    Y_intnoi_tgt = gather(Y_intnoi_tgt);
    Yf_int = gather(Yf_int);
    Y_interf_int = gather(Y_interf_int);
    Y_tarnoi_int = gather(Y_tarnoi_int);
    S = gather(S);
    S_tar = gather(S_tar);
    S_interf = gather(S_interf);
    S_intnoi = gather(S_intnoi);
    S_tarnoi = gather(S_tarnoi);
    a_target = gather(a_target);
    a_interf = gather(a_interf);
end

fprintf('[MVDR] MVDR processing complete.\n\n');

end

%% ========== MVDR Branch Function ==========
function [Y, Y_comp1, Y_comp2] = mvdr_branch(S, S_comp1, S_comp2, a_steer, ...
    config, Fbins, Tframes, f_start, f_limit, Nmic, branch_name)

Y = zeros(Fbins, Tframes, 'like', S(:,:,1));
Y_comp1 = zeros(size(Y));
Y_comp2 = zeros(size(Y));

for k = f_start:f_limit
    Xkf = squeeze(S(k,:,:)).';  % Nmic x Tframes
    Xkf_comp1 = squeeze(S_comp1(k,:,:)).';
    Xkf_comp2 = squeeze(S_comp2(k,:,:)).';
    
    if config.use_oracle_intnoi_cov && strcmp(branch_name, 'target-steered')
        Xkf_cov = Xkf_comp2;  % Use interference+noise
    elseif config.use_oracle_tarnoi_cov && strcmp(branch_name, 'interference-steered')
        Xkf_cov = Xkf_comp2;  % Use target+noise
    else
        Xkf_cov = Xkf;  % Use total signal
    end
    
    a = a_steer(:,k);
    if norm(a) < 1e-12
        continue;
    end
    
    if config.mvdr_use_time_varying
        w_last = zeros(Nmic, 1, 'like', a);
        for n = 1:Tframes
            if n == 1 || mod(n-1, config.mvdr_frame_stride) == 0
                t1 = max(1, n - floor(config.Mavg/2));
                t2 = min(Tframes, n + floor(config.Mavg/2));
                Xloc = Xkf_cov(:, t1:t2);
                
                Rxx = (Xloc * Xloc') / size(Xloc,2);
                
                if config.shrink_alpha > 0
                    mu = trace(Rxx) / Nmic;
                    Rxx = (1 - config.shrink_alpha) * Rxx + config.shrink_alpha * mu * eye(Nmic, 'like', Rxx);
                end
                
                Rxx = Rxx + config.epsilon * trace(Rxx)/Nmic * eye(Nmic, 'like', Rxx);
                w = Rxx \ a;
                denom = a' * w;
                if abs(denom) < 1e-12
                    w = zeros(Nmic, 1, 'like', a);
                else
                    w = w / denom;
                end
                w_last = w;
            end
            
            Y(k,n) = w_last' * Xkf(:,n);
            Y_comp1(k,n) = w_last' * Xkf_comp1(:,n);
            Y_comp2(k,n) = w_last' * Xkf_comp2(:,n);
        end
    else
        % Fast mode: one weight per frequency bin
        Rxx = (Xkf_cov * Xkf_cov') / Tframes;
        if config.shrink_alpha > 0
            mu = trace(Rxx) / Nmic;
            Rxx = (1 - config.shrink_alpha) * Rxx + config.shrink_alpha * mu * eye(Nmic, 'like', Rxx);
        end
        Rxx = Rxx + config.epsilon * trace(Rxx)/Nmic * eye(Nmic, 'like', Rxx);
        w = Rxx \ a;
        denom = a' * w;
        if abs(denom) < 1e-12
            w = zeros(Nmic, 1, 'like', a);
        else
            w = w / denom;
        end
        
        Y(k,:) = w' * Xkf;
        Y_comp1(k,:) = w' * Xkf_comp1;
        Y_comp2(k,:) = w' * Xkf_comp2;
    end
    
    if mod(k, config.mvdr_progress_step) == 0 || k == f_limit
        fprintf('[MVDR] %s: %.1f%% complete\n', branch_name, 100*k/f_limit);
    end
end

end

%% Helper function
function result = iif(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
