function overrides = mvdr_config_raw_datasets_transformer()
%% ============== Experiment Override: Raw Datasets Transformer Export ==============
% Batch export preset for raw_datasets. Saves only axis-free steered spectra
% so the output can be fed into the dual-channel transformer pipeline.

overrides = struct();
overrides.experiment_name = 'raw_datasets_transformer';
overrides.output_dir = '../output/raw_datasets_transformer';
overrides.figure_subdir = 'spectrums';
overrides.save_figures = true;
overrides.save_spectrums_only = true;
overrides.spectrum_remove_axes = true;
overrides.spectrum_output_format = 'png';
overrides.use_parallel_rir = false;

overrides.plot_style = struct();
overrides.plot_style.figure_bg = 'white';
overrides.plot_style.axes_bg = 'white';
overrides.plot_style.text_color = 'k';
overrides.plot_style.colormap_name = 'turbo';
overrides.plot_style.use_invert_hardcopy = false;
overrides.plot_style.line_width = struct('main', 1.5, 'medium', 2.0, 'thick', 2.5, 'thin', 1.2);
overrides.plot_style.marker_size = struct('main', 8, 'detail', 10);
overrides.plot_style.font_sz = struct(...
	'title', 30, ...
	'subtitle', 29, ...
	'label', 29, ...
	'subtile_title', 29, ...
	'legend', 28, ...
	'tick', 28, ...
	'small_tick', 27, ...
	'colorbar', 28, ...
	'colorbar_label', 28);

end
