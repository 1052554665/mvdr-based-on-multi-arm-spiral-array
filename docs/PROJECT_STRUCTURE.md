# 项目结构和使用指南

## 项目概述
这是基于多臂螺旋阵列的MVDR波束形成系统，使用实际信号和RIR（房间脉冲响应）进行处理。

## 目录结构

```
mvdr-based-on-multi-arm-spiral-array/
│
├── src/                              # 源代码目录
│   ├── mvdr_main.m                  # ★ 主程序（入口点）
│   ├── load_config.m                # 配置文件（所有参数集中在此）
│   ├── load_data_module.m           # 数据加载模块
│   ├── rir_generation_module.m      # RIR生成模块
│   ├── mvdr_processing_module.m     # MVDR处理核心模块
│   ├── diagnostics_and_visualization_module.m  # 诊断和可视化
│   ├── array_and_room_visualization_based_on_rir.m
│   ├── array_coordinate_visualization.m
│   └── sinewav.m
│
├── utils/                            # 工具脚本
│   └── xlsx_to_csv.m                # Excel转CSV转换工具
│
├── data/                             # 数据文件目录
│   ├── mic_positions.xlsx           # 麦克风位置（Excel格式）
│   ├── mic_positions.csv            # 麦克风位置（CSV格式）
│   └── audio/                        # 音频输入文件目录
│       ├── Normal_part92.wav         # 目标源音频
│       └── 振安1#反_part46.wav       # 干扰源音频
│
├── output/                           # 生成的输出文件目录
│   ├── figures/                      # PDF图表
│   │   ├── Microphone_Array_Geometry.pdf
│   │   ├── Room_and_Array_Layout.pdf
│   │   ├── Rxx_eigenspectrum_2kHz.pdf
│   │   ├── Beampattern_2kHz.pdf
│   │   ├── Performance_improvement.pdf
│   │   ├── PSD_fullband.pdf
│   │   └── ...
│   └── results/                      # 处理结果数据
│
├── docs/                             # 文档目录
│   ├── README.md                     # 项目说明
│   └── mvdr_of_real_signal_based_on_rir.md
│
└── .git/                             # Git版本控制
```

## 使用指南

- 输入音频文件放在data/audio
- Matlab安装`Parallel Computing Toolbox`
- 下载`RIR-Generator`并添加到MATLAB路径（https://www.audiolabs-erlangen.de/fau/professor/habets/resources/simulation）

### 1. 快速开始

```matlab
% 在 MATLAB 中运行：
cd src/
mvdr_main
```

这将：
- 加载配置参数
- 读取麦克风位置和音频文件
- 生成RIR（房间脉冲响应）
- 执行MVDR波束形成
- 生成诊断图表和性能指标

### 2. 配置参数

编辑 `src/load_config.m` 修改参数：

```matlab
% 示例：修改MVDR频率范围
config.mvdr_fmin_hz = 500;      % 改为500 Hz
config.mvdr_fmax_hz = 7000;     % 改为7 kHz

% 示例：修改房间参数
config.room_L = [6 5 3.5];      % 新的房间尺寸
config.beta = 0.2;              % 增加反射系数
```

### 3. 模块说明

| 模块 | 功能 | 关键参数 |
|-----|------|--------|
| `load_config.m` | 集中管理所有参数 | 音频、房间、MVDR、阵列 |
| `load_data_module.m` | 加载mic位置、音频文件 | `mic_pos_file`, 音频路径 |
| `rir_generation_module.m` | 生成RIR | `beta`, `room_L`, `nsample` |
| `mvdr_processing_module.m` | MVDR波束形成 | `mvdr_fmin/fmax`, `Mavg`, `epsilon` |
| `diagnostics_and_visualization_module.m` | 诊断和绘图 | `save_figures`, `output_dir` |

### 4. 主要参数解释

#### 音频参数
- `c`: 声速 (340 m/s)
- `fs`: 采样率 (16 kHz)
- `INR_dB`: 干扰与目标信号比例

#### 房间参数
- `room_L`: 房间尺寸 [长, 宽, 高] (米)
- `beta`: 墙壁反射系数 (0=仅直达声)
- `s_target`: 目标源位置
- `s_interf`: 干扰源位置

#### MVDR参数
- `mvdr_fmin_hz` / `mvdr_fmax_hz`: 处理频率范围
- `Mavg`: 协方差矩阵平均窗口大小（帧数）
- `epsilon`: 对角线加载强度
- `shrink_alpha`: 协方差矩阵收缩因子

### 5. 输出文件

运行完成后，在 `output/figures/` 目录下会生成：

**可视化PDF：**
- `Microphone_Array_Geometry.pdf` - 2D麦克风阵列布局
- `Room_and_Array_Layout.pdf` - 3D房间与声源布局
- `Rxx_eigenspectrum_2kHz.pdf` - 协方差矩阵特征值谱
- `Beampattern_2kHz.pdf` - 2kHz处的波束图案
- `Performance_improvement.pdf` - MVDR前后性能对比
- `PSD_fullband.pdf` - 全频段功率谱密度
- `PSD_4kHz.pdf` - 4kHz处的功率谱密度

**控制台输出：**
- 输入SINR/SIR
- 输出SINR/SIR
- SNR改进量（dB）
- 波束方向性指标

### 6. 核心性能指标

程序计算并显示以下性能指标：

```
INPUT SNR:        处理前的信噪比
OUTPUT SNR:       处理后的信噪比  
SNR GAIN:         SNR改进量（dB）

INPUT ISR:        处理前的干扰比
OUTPUT ISR:       处理后的干扰比
ISR GAIN:         ISR改进量（dB）
```

### 7. 常见自定义

#### 修改源位置
```matlab
% 在 load_config.m 中
config.s_target = [2.5 1 3];    % 目标源 [x, y, z]
config.s_interf = [2.5 3 3];    % 干扰源 [x, y, z]
```

#### 修改阵列中心
```matlab
config.array_center = [2.5 2 1.5];  % 阵列中心位置
```

#### 启用时变MVDR权重
```matlab
config.mvdr_use_time_varying = true;  % 更慢但更自适应
config.mvdr_frame_stride = 1;         % 每帧更新权重
```

#### 使用GPU加速
```matlab
config.use_gpu = true;          % 需要GPU和并行计算工具箱
config.use_parallel_rir = true; % RIR并行生成
```

## 技术细节

### MVDR算法
- 目标指向分支：保留目标，抑制干扰+噪声
- 干扰指向分支：保留干扰，抑制目标+噪声
- 对角线加载：增强稳定性（参数：`epsilon`）
- 协方差收缩：进一步稳定性改进（参数：`shrink_alpha`）

### 立体声处理
- STFT分析 → 频率和时间帧
- 每个频率单独计算MVDR权重
- 逆STFT重建时域信号

## 故障排查

| 问题 | 原因 | 解决方案 |
|-----|------|--------|
| 音频文件找不到 | 路径配置错误 | 检查 `load_config.m` 中的文件路径 |
| 内存不足 | 信号过长或FFT长度太大 | 减小 `config.Nfft` 或 `config.nsample` |
| 波束方向错误 | 麦克风位置或源位置配置错误 | 验证 `mic_positions.xlsx` 和源位置坐标 |
| MVDR权重数值不稳定 | 对角线加载不足 | 增加 `config.epsilon` |

## 扩展建议

1. **多源处理**：修改 `mvdr_processing_module.m` 支持多个目标或干扰
2. **自适应处理**：启用 `mvdr_use_time_varying = true`
3. **频率加权**：在MVDR循环中添加频率相关的权重
4. **性能优化**：使用GPU加速或并行处理


