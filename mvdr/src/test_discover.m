raw_root = '../raw_datasets';
root_entries = dir(raw_root);
fprintf('All entries:\n');
for k=1:numel(root_entries)
    fprintf('  [%d] name=%s isdir=%d\n', k, root_entries(k).name, root_entries(k).isdir);
end

root_entries = root_entries([root_entries.isdir]);
fprintf('\nAfter isdir filter:\n');
for k=1:numel(root_entries)
    fprintf('  [%d] %s\n', k, root_entries(k).name);
end

root_entries = root_entries(~ismember({root_entries.name}, {'.', '..'}));
fprintf('\nAfter removing . and ..:\n');
for k=1:numel(root_entries)
    fprintf('  [%d] %s\n', k, root_entries(k).name);
end

known_splits = {'train', 'val', 'test'};
fprintf('\nChecking strcmpi:\n');
for k=1:numel(root_entries)
    entry_name = root_entries(k).name;
    match = any(strcmpi(entry_name, known_splits));
    fprintf('  %s -> match=%d\n', entry_name, match);
end
