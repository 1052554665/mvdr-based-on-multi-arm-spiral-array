- 读取目录
- 写一个外层脚本，遍历所有配对
- 每个目标类都和 normal 类随机混合

调用方式
```matlab
batch_mvdr_raw_datasets('../raw_datasets', 'real_signal_dcbias_4k', '../output/raw_datasets_mvdr')
```

# 数据集产生方式
**输出对应关系**
- 目标增强、干扰抑制：`Yf_tgt`，以及它的分量 `Y_tar_tgt` / `Y_intnoi_tgt`
- 干扰增强、目标抑制：`Yf_int`，以及它的分量 `Y_interf_int` / `Y_tarnoi_int`

这些都已经在 mvdr_processing_module.m 和 diagnostics_and_visualization_module.m 里生成了，不需要改 beamformer 公式，只需要把数据加载方式改成“目录批处理 + 每次取一对”。

**推荐的数据组织方式**
- `train/`
  - `fault_1/ ... fault_9/`  9 种目标故障信号
  - `normal/`  1 种干扰信号
- `val/`
  - 同上
- `test/`
  - 同上

每个音频都尽量保持 1 秒，采样率和 `config.fs` 一致。当前 `load_data_module` 会自动做单声道、重采样、截断到相同长度；但如果你希望时频图尺寸稳定，最好还是预先统一成 1 秒。

**实际加载策略**
1. 先为每个 split 建两个列表：
   - 目标文件列表：9 个故障类里的所有 `.wav`
   - 干扰文件列表：normal 类里的所有 `.wav`
2. 训练/验证/测试时，按索引或随机方式配对：
   - `target_file = fault_x/sample_k.wav`
   - `interf_file = normal/sample_j.wav`
3. 每次只运行一次 `mvdr_main` 的那套流程：
   - `config.target_audio = target_file`
   - `config.interf_audio = interf_file`
   - 然后调用 `load_data_module -> rir_generation_module -> mvdr_processing_module -> diagnostics...`
4. 这样就会得到一对输出时频图：
   - 目标增强谱
   - 干扰增强谱

**批量跑整个数据集**
写一个外层脚本，遍历所有配对，循环更新 `config.target_audio` 和 `config.interf_audio`。

如果目标是“每个目标类都和 normal 类随机混合”，这个方式最直接。若想更稳一点，也可以先把配对列表写成 CSV，再按 CSV 逐行读入。

**关键点**
- 这个 MVDR 版本一次只处理一个目标源和一个干扰源，不支持“9 个目标同时作为一个输入”。
- 9 类目标的作用是“数据集标签/样本来源”，不是 beamformer 内部同时混合的 9 个源。
- 最终要的“增强目标、抑制干扰”的时频图，直接看 `Yf_tgt`；“增强干扰、抑制目标”的图，看 `Yf_int`。
- 诊断图和输出 PDF 已经在 diagnostics_and_visualization_module.m 里保存到 `output/.../figures/`。


