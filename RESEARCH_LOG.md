# 默认配置 + 实验覆盖

配置入口变成了“默认配置 + 实验覆盖”模式：
- 默认值在 `mvdr_default_config.m`，加载器在 `load_config.m` 里先读默认配置，再按实验名叠加覆盖文件。
- 示例实验：2 kHz/4 kHz 的 `mvdr_config_demo_2k_4k.m`，以及当前真实信号基线的 `mvdr_config_real_signal_dcbias_4k.m`。
- 主入口显式选择实验名，见 `mvdr_main.m`。
- 图形输出目录也不再写死，改成走 `config.figure_subdir`，见 `diagnostics_and_visualization_module.m` 和 `mvdr_default_config.m`。


# 数据加载
数据加载：
- 2kHz + 4kHz
- DCBias + 4kHz
- 公开数据集