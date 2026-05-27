function config = load_config(experiment_name)
%% ============== MVDR Configuration Loader ==============
% Loads the shared default configuration and applies experiment-specific overrides.

if nargin < 1 || isempty(experiment_name)
    experiment_name = 'real_signal_dcbias_4k';
end

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(script_dir, 'configs')));

set(groot, ...
    'defaultAxesFontName', 'Times New Roman', ...
    'defaultTextFontName', 'Times New Roman', ...
    'defaultLineLineWidth', 1.2);

config = mvdr_default_config();
config.experiment_name = experiment_name;

override_fn = matlab.lang.makeValidName(['mvdr_config_' experiment_name]);
override_path = which([override_fn '.m']);

if ~isempty(override_path)
    overrides = feval(override_fn);
    config = mvdr_merge_config(config, overrides);
    config.experiment_name = experiment_name;
    fprintf('[Config] Loaded experiment configuration: %s\n', experiment_name);
else
    fprintf('[Config] Using default configuration only (no override found for "%s").\n', experiment_name);
end

fprintf('[Config] Configuration loaded.\n');

end
