>真实信号RIR实验设置

- 修改图片输出路径：在diagnostics_and_visualization_module.m中，
    ```matlab
    figures_dir = fullfile(config.output_dir, 'figures');
    ```
- 修改音频文件路径：load_config.m中，使用了真实信号RIR实验设置的音频文件作为目标信号和干扰信号。

    ```matlab
    config.target_audio = '../data/audio/Loosen1_60.wav';
    config.interf_audio = '../data/audio/Normal_part92.wav';
    ```
- figure(207)
    ```
    h = legend('Target-steered', 'Interference-steered', 'Location', 'best');

    ```
- diagnostics_and_visualization_module.m中设置全部字体大小，以满足IEEE出版标准。
- 六宫图单独设置