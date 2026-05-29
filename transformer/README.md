# Docker挂载
按你当前项目配置，最省事就是用 Docker Compose，因为挂载已经在 docker-compose.yaml 里写好了。
1. 克隆项目  
在你希望存放代码的目录执行：
    ```bash
    git clone 你的仓库地址 transformer  
    cd transformer
    ```
2. 确认本地目录结构
确保至少有这些目录（没有就创建）：
    ```bash
    mkdir -p data/speech_enhancement/clean  
    mkdir -p data/speech_enhancement/noise  
    mkdir -p data/speech_enhancement/noisy  
    mkdir -p experiments  
    mkdir -p enhancement_report
    ```
3. 启动容器并自动挂载
直接执行：
    ```bash
    docker compose up --build
    ```
当前配置会自动把本机目录挂载到容器内：  
- ./data -> /app/data  
- ./experiments -> /app/experiments  
- ./enhancement_report -> /app/enhancement_report

对应配置可见 docker-compose.yaml。

4. 验证挂载是否生效  
另开终端执行：
    ```bash
    docker exec -it dual_transformer bash  
    ls /app/data  
    ls /app/experiments  
    ls /app/enhancement_report
    ```
如果能看到本机同名目录内容，说明挂载成功。

5. 停止容器  
在 compose 终端里按 Ctrl+C，或执行：
```bash
docker compose down
```
补充说明：  
当前容器默认执行命令来自 docker-compose.yaml，即运行训练脚本。基础镜像和依赖安装逻辑在 Dockerfile。


# 项目介绍
这份代码在`main.py`实现的是一个双分支 Transformer 语音增强训练与评估脚本，目标是：
输入带噪语音频谱$x_s$和噪声参考频谱$x_n$，输出增强语音和时频掩码 mask。

NOTE：
- 使用远程服务器训练，结果保存在远程。`Remote Host`查看文件保存位置
- 训练中的报错、package的安装环境也要在终端中转换为远程服务器解决。
- `num_frames = (audio_duration × sample_rate - n_fft) // hop_length + 1`， 根据音频长度调整input_size的时间维度。`input_size': (128, 64)`的第二个参数
- `num_workers = 0`
- 新的数据集添加到本地并重命名后，远程并没有同步改名。所以需要在远程服务器上也进行重命名，或者直接在远程服务器上添加数据集并重命名。


根据提供的文件内容，"transformer"项目的功能如下：

1. 主要功能：
- 实现了一个双分支Transformer模型用于语音增强
- 输入带噪语音频谱和噪声参考频谱，输出增强语音和时频掩码
- 提供了多种噪声抑制模式：自适应、激进式和频域感知

2. 核心组件：
- PatchEmbedding模块
- TransformerEncoder编码器
- CrossAttentionFusion融合层
- Decoder解码器
- DualBranchTransformer主模型

3. 功能特点：
- 支持三种噪声抑制模式，默认使用频域感知模式
- 增加了水平条纹检测机制（horizontal_detector）
- 采用复合损失函数：增强MSE + 掩码重建MSE + 稀疏正则
- 提供完整的训练、验证和推理流程

4. 数据处理：
- 支持随机配对噪声合成数据
- 自动生成训练/验证/测试集划分
- 支持Mel频谱构建和尺寸对齐

5. 评估与可视化：
- 统计MSE、SNR改善等指标
- 生成四图对比（clean/noisy/enhanced/mask）
- 输出增强结果为.npz格式
- 生成图像+文本报告

6. 配置管理：
- 使用YAML文件配置参数
- 支持命令行参数覆盖配置
- 提供完整的数据集结构模板

7. 环境要求：
- 基于PyTorch框架
- 需要安装numpy、matplotlib、librosa等依赖库
- 提供Docker容器化支持


# 执行训练
## 本机直接训练（推荐先用这个确认流程）

先进入项目根目录
```bash
cd transformer
```
然后执行：
```bash
python scripts/train.py --config configs/train.yaml
```
使用评估入口基于已有 checkpoint 生成报告（不会重新训练）：
  ```bash
  cd transformer
  python scripts/evaluate.py --config configs/train.yaml --checkpoint experiments/train/best_model.pth --output-dir enhancement_report --num-samples 20
  ```



## Docker 里训练（已配置挂载）
在项目根目录执行：
```bash
docker compose up --build
```
默认会执行训练命令，配置来自 docker-compose.yaml。
容器内训练脚本由 Dockerfile 和 docker-compose.yaml 指定。

1. 常见检查点
- 配置文件路径是否存在：train.yaml
- 训练脚本是否存在：train.py
- 数据目录是否有内容：data/speech_enhancement/clean 和 data/speech_enhancement/noise


# 模型结构（Dual-Branch Transformer）

- PatchEmbedding：把 2D 时频图切 patch 并映射到 token。 
- TransformerEncoder：共享编码器分别处理语音分支和噪声分支。
- CrossAttentionFusion：用交叉注意力和门控机制抑制噪声特征。
- Decoder：把 token 解码为时频掩码并上采样回原尺寸。
- DualBranchTransformer：输出 enhanced = mask * noisy_magnitude 与 mask。

## CrossAttentionFusion——3种抑制模式
新增3种抑制模式（默认使用frequency_aware）：

|模式|机制|适用场景
| ---|---|---
|adaptive|原软门控 $z_s - g*z_cross$|基线对比
|aggressive|双重抑制：放大干扰后减去 + 增强差异|强噪声
|frequency_aware|频域感知：检测水平条纹 → 强化识别 → 激进去除|去除噪声水平条纹（推荐）

频域感知增加了非对称卷积检测机制

水平方向（长核 kernel_size=5）
horizontal_detector → 检测频域连续性（噪声条纹特征）

垂直方向（短核 kernel_size=3）  
vertical_detector → 检测语音的垂直结构（谐波/共振峰）

条纹得分 = 水平强响应 AND 垂直弱响应
stripe_score = concat(h_response, 1 - v_response)
freq_attention = stripe_enhancer(stripe_score)


## 数据处理与数据集
* SpeechEnhancementDataset 支持两种模式： 
  * 真实数据目录（clean/, noise/, 可选 noisy/）
  * 无数据时自动生成合成样本。
* 提供音频加载、noisy 合成（按随机 SNR）、Mel 频谱构建、尺寸对齐、双通道特征（mel + delta）。

与图像分类不同，语音增强有两种数据模式：

| 模式     | Clean-Noise关系 |      实现方式                    |     适用场景                     | 说明                                                  |优点|
| -------- | --------------- |--------------------------|--------------------------|-----------------------------------------------------|---|
| 固定配对 | 一一对应        |        clean[i] + noise[i]       |     特定房间/设备                     | 每个clean有指定的noise（如特定房间脉冲响应）     |模拟真实环境|
| 随机混合 | 无需对应        |           clean[i] + noise[j]    |     通用噪声抑制                     | 任意clean + 任意noise组合（更常见） |数据多样性高|

本代码采用的是随机混合模式，这样可以
- 增加数据多样性（N_clean × N_noise种组合），如果有100个clean和50个noise，可生成5000种组合
- 模型学习通用的噪声抑制能力而非记忆特定配对
- 避免过拟合特定噪声类型，提升泛化性能

数据集划分策略：
```text
总数据 (100%)
├── 训练集 (70%) - 用于模型学习
├── 验证集 (15%) - 用于调参和早停
└── 测试集 (15%) - 用于最终评估（完全独立）
```
同时分别独立划分clean和noise文件

| 维度       | 说明                                                 |
| ---------- | ---------------------------------------------------- |
| Clean集合  | 独立划分为 train_clean / val_clean / test_clean      |
| Noise集合  | 独立划分为 train_noise / val_noise / test_noise      |

优势：
- 防止数据泄露：测试集的clean不会与训练集的noise配对
- 灵活处理不等数量：clean和noise可以有不同的文件数
- 保证评估公平性：测试时只使用test_clean + test_noise的组合


## 训练与验证
- compute_loss：复合损失（增强 MSE + 掩码重建 MSE + 稀疏正则）。
- train_one_epoch：标准训练循环（反向传播、梯度裁剪、日志打印）。
- validate：验证集损失评估。
- main：完整训练流程（配置、DataLoader、优化器、调度器、best/checkpoint 保存）。

## 推理与工具函数
- test_inference：随机输入做 shape/value 范围检查。
- prepare_dataset_structure：生成数据目录模板。
- generate_and_save_noisy_audio：批量生成带噪音频并写入 noisy。

## 评估与可视化
- evaluate_speech_quality：统计 MSE、SNR 前后与提升（PESQ/STOI接口预留，但当前未实际填充计算值）。
- save_enhanced_audio：保存增强结果为 .npz 频谱数据。
- visualize_spectrogram_comparison：生成 clean/noisy/enhanced/mask 四图对比。
- generate_enhancement_report：输出图像+文本报告。

## 数据目录路径
```text
data/speech_enhancement/
├── clean/                # 干净语音文件
│   ├── speech_001.wav
│   ├── speech_002.wav
│   └── ...
├── noise/                # 噪声文件
│   ├── noise_001.wav
│   ├── noise_002.wav
│   └── ...
└── noisy/                # （可选）预生成的带噪语音
    ├── noisy_001.wav
    └── ...
```

## 指标评估
| 指标              | 说明           | 范围   | 越好     |
| ----------------- | -------------- | ------ | -------- |
| SNR Improvement   | 信噪比提升     | dB     | ↑越高越好 |
| MSE               | 谱图均方误差   | 0-∞    | ↓越低越好 |
| PESQ              | 感知语音质量\* | 1-4.5  | ↑越高越好 |
| STOI              | 短时客观可懂度\* | 0-1   | ↑越高越好 |


| 对比项               | 好的表现                     | 差的表现               |
| -------------------- | ---------------------------- | ---------------------- |
| Noisy vs Clean       | -                            | 大量红色噪声区域       |
| Enhanced vs Clean   | 谱图结构相似                 | 过度平滑/失真          |
| Mask分布             | 噪声区≈0，语音区≈1           | 全0或全1               |
| SNR改善              | +5dB以上                     | <2dB或负值             |

颜色说明：
- 谱图 (inferno colormap): 红色=高能量，黑色=低能量
- Mask (viridis colormap): 黄色=保留(≈1)，紫色=抑制(≈0)

视觉检查：
- Noisy谱图：看到明显的水平条纹（周期性噪声）或随机斑点（宽带噪声）
- Enhanced谱图：噪声纹理消失，语音共振峰（横向条纹）更清晰
- Mask：噪声区域显示深紫色（被抑制），语音区域显示黄色（保留）

