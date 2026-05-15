>2kHz和4kHz的RIR实验设置

- 修改图片输出路径：在diagnostics_and_visualization_module.m中，
    ```matlab
    figures_dir = fullfile(config.output_dir, 'demo');
    ```
- 修改音频文件路径：load_config.m中，使用2kHz和4kHz的正弦波作为目标信号和干扰信号。

    ```matlab
    config.target_audio = '../data/audio/sine_wave_2k.wav';
    config.interf_audio = '../data/audio/sine_wave_4k.wav';
    ```
- figure(207)
    ```
    h = legend('2kHz-steered', '4kHz-steered', 'Location', 'best');

    ```
- 字号设置调整：在diagnostics_and_visualization_module.m中设置全部字体大小，以满足IEEE出版标准。
- 六宫图单独设置
