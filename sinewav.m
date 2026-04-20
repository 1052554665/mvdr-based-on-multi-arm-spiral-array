% Sine Wave Generator
% Parameters
fs = 44100;        % Sampling frequency (Hz)
duration = 2;      % Duration (seconds)
frequency = 4;   % Sine wave frequency (Hz)
amplitude = 0.8;   % Amplitude (0 to 1)

% Generate time vector
t = linspace(0, duration, fs * duration);

% Generate sine wave
sineWave = amplitude * sin(4 * pi * frequency * t);

% Save as WAV file
audiowrite('sine_wave_4k.wav', sineWave, fs);

% Plot the waveform
plot(t, sineWave);
xlabel('Time (s)');
ylabel('Amplitude');
title('Sine Wave');
grid on;