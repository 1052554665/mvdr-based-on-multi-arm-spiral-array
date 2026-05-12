%% ============== RIR Generation Module ==============
% Generates room impulse responses for target and interference sources

function [h_target, h_interf, X_target, X_interf, X_noisy] = rir_generation_module(config, x_target, x_interf, mic_pos, Nt, Nmic)

fprintf('\n[RIR Generation] Starting RIR generation...\n');

%% 1. Determine RIR length
nsample_use = config.nsample;

if config.beta == 0
    % Direct sound only - can use shorter RIR
    max_dist = max([vecnorm(mic_pos - config.s_target, 2, 2); 
                    vecnorm(mic_pos - config.s_interf, 2, 2)]);
    nsample_direct = max(256, ceil(max_dist / config.c * config.fs) + 128);
    nsample_use = min(config.nsample, nsample_direct);
    fprintf('[RIR] Direct sound only (beta=0). Using shortened RIR: %d samples\n', nsample_use);
else
    fprintf('[RIR] Using full RIR length: %d samples\n', nsample_use);
end

%% 2. Initialize RIR matrices
h_target = zeros(Nmic, nsample_use);
h_interf = zeros(Nmic, nsample_use);

%% 3. Setup parallel computation if available
use_parallel = config.use_parallel_rir && license('test', 'Distrib_Computing_Toolbox');

if use_parallel
    p = gcp('nocreate');
    if isempty(p)
        try
            parpool('Processes');
            fprintf('[RIR] Parallel pool started\n');
        catch ME
            warning('RIR:pool', 'Failed to start parallel pool: %s. Using serial computation.\n', ME.message);
            use_parallel = false;
        end
    end
end

%% 4. Generate RIRs
fprintf('[RIR] Generating RIRs for %d microphones...\n', Nmic);
tic;

if use_parallel
    parfor m = 1:Nmic
        rm = mic_pos(m,:);
        ht = rir_generator(config.c, config.fs, rm, config.s_target, config.room_L, ...
            config.beta, nsample_use, config.mtype, config.order, config.dim, ...
            config.orientation, config.hp_filter);
        hi = rir_generator(config.c, config.fs, rm, config.s_interf, config.room_L, ...
            config.beta, nsample_use, config.mtype, config.order, config.dim, ...
            config.orientation, config.hp_filter);
        
        % Ensure row vectors
        if iscolumn(ht), ht = ht.'; end
        if iscolumn(hi), hi = hi.'; end
        
        h_target(m,:) = ht;
        h_interf(m,:) = hi;
    end
else
    for m = 1:Nmic
        rm = mic_pos(m,:);
        ht = rir_generator(config.c, config.fs, rm, config.s_target, config.room_L, ...
            config.beta, nsample_use, config.mtype, config.order, config.dim, ...
            config.orientation, config.hp_filter);
        hi = rir_generator(config.c, config.fs, rm, config.s_interf, config.room_L, ...
            config.beta, nsample_use, config.mtype, config.order, config.dim, ...
            config.orientation, config.hp_filter);
        
        % Ensure row vectors
        if iscolumn(ht), ht = ht.'; end
        if iscolumn(hi), hi = hi.'; end
        
        h_target(m,:) = ht;
        h_interf(m,:) = hi;
        
        if mod(m, max(1, Nmic/5)) == 0
            fprintf('[RIR]   %d/%d microphones processed...\n', m, Nmic);
        end
    end
end

elapsed = toc;
fprintf('[RIR] RIR generation completed in %.2f seconds\n', elapsed);

%% 5. Convolve signals with RIRs
fprintf('[RIR] Convolving signals with RIRs...\n');

X_target = zeros(Nt, Nmic);
X_interf = zeros(Nt, Nmic);
noise = zeros(Nt, Nmic);

for m = 1:Nmic
    X_target(:,m) = conv(x_target, h_target(m,:), 'same');
    X_interf(:,m) = conv(x_interf, h_interf(m,:), 'same');
end

% Combine signals (no synthetic noise added - using loaded signals directly)
X_noisy = X_target + X_interf;

fprintf('[RIR] Signal shapes: X_target [%d x %d], X_interf [%d x %d]\n', ...
    size(X_target,1), size(X_target,2), size(X_interf,1), size(X_interf,2));

%% 6. Compute input SNR/SIR metrics
fprintf('[RIR] Computing input SNR/SIR metrics...\n');

ref_mic = config.ref_mic;
x_tar_in = X_target(:, ref_mic);
x_int_in = X_interf(:, ref_mic);

P_tar_in = mean(x_tar_in.^2);
P_int_in = mean(x_int_in.^2);
P_intnoi_in = P_int_in + eps;  % No synthetic noise

SNR_in_dB = 10*log10(P_tar_in / P_intnoi_in);
SIR_in_dB = 10*log10(P_tar_in / P_int_in);

fprintf('\n===== INPUT SNR (Reference Mic %d) =====\n', ref_mic);
fprintf('  Target power:        %.4f\n', P_tar_in);
fprintf('  Interference power:  %.4f\n', P_int_in);
fprintf('  Input SINR:          %.2f dB\n', SNR_in_dB);
fprintf('  Input SIR:           %.2f dB\n', SIR_in_dB);
fprintf('==========================================\n\n');

fprintf('[RIR] RIR generation module complete.\n\n');

end
