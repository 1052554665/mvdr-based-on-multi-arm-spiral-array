function config = mvdr_merge_config(config, overrides)
%% ============== MVDR Config Merge Helper ==============
% Recursively applies override fields onto the base configuration.

if nargin < 2 || isempty(overrides)
    return;
end

if ~isstruct(overrides)
    error('MVDR:Config', 'Overrides must be a struct.');
end

override_fields = fieldnames(overrides);
for idx = 1:numel(override_fields)
    field_name = override_fields{idx};
    override_value = overrides.(field_name);

    if isstruct(override_value) && isfield(config, field_name) && isstruct(config.(field_name))
        config.(field_name) = mvdr_merge_config(config.(field_name), override_value);
    else
        config.(field_name) = override_value;
    end
end

end
