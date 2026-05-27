function overrides = mvdr_config_multi_beta()
%% ============== Experiment Override: Multi-Beta Analysis ==============
% Configuration for the MVDR multi-beta parameter sweep analysis

overrides = struct();
overrides.experiment_name = 'multi_beta_analysis';

% The multi-beta analysis writes comparison figures and CSVs to this folder
overrides.output_dir = '../output/multi_beta';
overrides.figure_subdir = 'figures';

% Plot style tuned for multi-panel comparison figures (publication-ready)
overrides.plot_style = struct();
overrides.plot_style.figure_bg = 'white';
overrides.plot_style.axes_bg = 'white';
overrides.plot_style.text_color = 'k';
overrides.plot_style.colormap_name = 'parula';
overrides.plot_style.use_invert_hardcopy = false;

% Line and marker widths used across the analysis
overrides.plot_style.line_width = struct('main', 2.2, 'medium', 1.8, 'thick', 2.8, 'thin', 1.1);
overrides.plot_style.marker_size = struct('main', 10, 'detail', 7);

% Font sizes chosen to match the sizes used in multi-beta analysis figures
overrides.plot_style.font_sz = struct( ...
    'title', 24, ...
    'subtitle', 22, ...
    'label', 22, ...
    'subtile_title', 22, ...
    'legend', 20, ...
    'tick', 19, ...
    'small_tick', 17, ...
    'colorbar', 20, ...
    'colorbar_label', 20);

% Optional legend overrides (if the analysis calls them)
overrides.legend_mvdr = {'Before', 'After'};
overrides.legend_psd = {'Before', 'After'};

overrides.eigen_plot_names = {'2kHz only', 'Full-band (2k+4k)'};

end
