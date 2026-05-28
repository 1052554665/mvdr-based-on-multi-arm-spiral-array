Optimized tool selection我先创建一个三步任务清单，然后扫描 transformer 目录并总结其功能。

Created 3 todos

接下来我将自动扫描 transformer 文件夹以收集主要文件和模块信息。

Search complete: Explore transformer

Searched codebase for "transformer folder analysis entrypoints configs model definitions dataset classes trainers utilities visualize modules requirements", no results

Searched for files matching `**/transformer**`, no matches

Searched for files matching `**/scripts/**`, 4 matches

Searched for files matching `**/src/**`, 29 matches

Searched for text `transformer`, 48 results

Searched for files matching `**/*.yaml`, 3 matches

Searched for files matching `**/*.py`, 18 matches

Read [](file:///home/yangchen/git_clone/mvdr-based-on-multi-arm-spiral-array/transformer/requirements.txt)

Read [](file:///home/yangchen/git_clone/mvdr-based-on-multi-arm-spiral-array/transformer/Dockerfile)

Read [](file:///home/yangchen/git_clone/mvdr-based-on-multi-arm-spiral-array/transformer/docker-compose.yaml)

Read [](file:///home/yangchen/git_clone/mvdr-based-on-multi-arm-spiral-array/transformer/src/models/dual_branch_transformer.py)

Read [](file:///home/yangchen/git_clone/mvdr-based-on-multi-arm-spiral-array/transformer/src/utils/config.py)

Read [](file:///home/yangchen/git_clone/mvdr-based-on-multi-arm-spiral-array/transformer/src/datasets/speech_enhancement.py)

Read [](file:///home/yangchen/git_clone/mvdr-based-on-multi-arm-spiral-array/transformer/src/trainers/workflow.py)

Read [](file:///home/yangchen/git_clone/mvdr-based-on-multi-arm-spiral-array/transformer/src/utils/train_eval.py)

Read [](file:///home/yangchen/git_clone/mvdr-based-on-multi-arm-spiral-array/transformer/configs/default.yaml)

Read [](file:///home/yangchen/git_clone/mvdr-based-on-multi-arm-spiral-array/transformer/configs/train.yaml)

Read [](file:///home/yangchen/git_clone/mvdr-based-on-multi-arm-spiral-array/transformer/main.py)

Read [](file:///home/yangchen/git_clone/mvdr-based-on-multi-arm-spiral-array/transformer/src/losses/speech_enhancement_loss.py)

现在我把任务清单标为完成，并返回简短中文功能总结。

Completed: *列出 `transformer` 目录文件* (1/3)

我已扫描 transformer 目录并整理出项目实现的功能，以下为简洁总结与关键文件参考。

**总体概述**:
- 该子项目实现了一个基于双分支 Transformer 的语音增强流水线，包含数据准备、模型定义、训练、评估、可视化和 Docker 容器化支持。

**关键功能点**:
- **模型**: 双分支 Transformer（语音分支 + 噪声参考分支），主要实现文件：src/models/dual_branch_transformer.py。输出增强谱与时频掩码（mask），支持 `frequency_aware` / `aggressive` / `adaptive` 多种抑制模式。
- **数据集**: 支持真实数据目录（clean/noise/可选 noisy）或合成噪声混合；Mel谱与 delta 特征，尺寸对齐，代码在 src/datasets/speech_enhancement.py。
- **训练流程**: 构建 DataLoader、训练/验证循环、学习率调度、检查点保存，主训练逻辑在 src/trainers/workflow.py；训练入口脚本为 scripts/train.py（也由 main.py 调用）。
- **损失与评估**: 复合损失（增强 MSE + 掩码重建 + 稀疏项）在 src/losses/speech_enhancement_loss.py；评估／可视化（PESQ/STOI 可选、SNR/MSE 指标、对比图生成、报告生成）在 src/utils/train_eval.py 与 scripts/evaluate.py。
- **工具与实用脚本**: 快速自检脚本 scripts/smoke_check.py；数据准备/生成工具在 `speech_enhancement.py` 中（`prepare_dataset_structure`、`generate_and_save_noisy_audio`）。
- **配置与可复现性**: YAML 配置文件：configs/default.yaml 与 configs/train.yaml；配置加载位于 src/utils/config.py。
- **环境与容器化**: 提供 `Dockerfile` 与 `docker-compose.yaml`，并在 README.md 中说明了挂载与运行（支持直接在容器内训练）。
- **依赖**: 基于 PyTorch，常用依赖包括 numpy、librosa、scipy、matplotlib、soundfile、pystoi/pesq（可选）。查看 requirements.txt 与 README 获取细节。

**如何快速验证/运行**:
- 本地训练（示例）:
  ```bash
  cd transformer
  python scripts/train.py --config configs/train.yaml
  ```
- 容器化运行:
  ```bash
  docker compose up --build
  ```
- 快速烟雾测试:
  ```bash
  python scripts/smoke_check.py
  ```