# MVDR-Based Acoustic Fault Diagnosis on Multi-Arm Spiral Array

> **Pipeline**: Multi-arm spiral array beamforming (MVDR) → Dual-Branch Transformer speech enhancement → Spectrogram-based fault classification

This project implements an end-to-end acoustic fault diagnosis system for power equipment, combining MVDR beamforming on a custom multi-arm spiral microphone array, deep learning-based speech enhancement, and multi-model classification.

```mermaid
flowchart LR
    A[Raw Audio<br/>Multi-Arm Spiral Array] --> B[MVDR Beamforming<br/>MATLAB]
    B --> C[Enhanced Spectrogram<br/>Dual-Branch Transformer<br/>PyTorch]
    C --> D[Fault Classification<br/>SVM/RF/MLP/CNN]
    D --> E[Diagnosis Result<br/>DCBias / Harmonic / Loosen / PartialDischarge]
```

---

## Problem

Acoustic fault diagnosis in power equipment faces two fundamental challenges:

1. **Spatial interference**: In industrial environments, target acoustic signals are contaminated by strong background noise and competing sound sources. Single-microphone approaches cannot spatially discriminate the target source from interferers, resulting in poor Signal-to-Interference-plus-Noise Ratio (SINR).

2. **Spectrogram-level noise**: Even after spatial filtering, residual noise manifests as horizontal stripes and random artifacts in time-frequency representations, degrading the performance of downstream classification models.

**Goal**: Achieve robust, high-accuracy fault classification (DC Bias, Harmonic, Loosen, Partial Discharge) from acoustic signals captured in noisy, reverberant environments.

---

## Method

A three-stage pipeline combining **spatial filtering**, **deep learning enhancement**, and **multi-model classification**:

```mermaid
flowchart LR
    subgraph Stage 1["Stage 1: Spatial Filtering"]
        A[Multi-Arm<br/>Spiral Array] --> B[Wideband MVDR<br/>Beamforming]
    end
    subgraph Stage 2["Stage 2: Spectrogram Enhancement"]
        C[Dual-Branch<br/>Transformer] --> D[Frequency-Aware<br/>Noise Suppression]
    end
    subgraph Stage 3["Stage 3: Fault Classification"]
        E[SVM / RF / LDA] --> F[ Diagnosis]
        G[MLP / CNN] --> F
    end
    B --> C
    D --> E
    D --> G
```

| Stage | Technique | Key Innovation |
|-------|-----------|----------------|
| **1. Spatial Filtering** | Wideband MVDR beamforming on custom multi-arm spiral array | Fraunhofer far-field steering vector construction; multi-$\beta$ parameter optimization; RIR-based propagation modeling |
| **2. Spectrogram Enhancement** | Dual-Branch Transformer with frequency-aware CrossAttentionFusion | Asymmetric convolution for horizontal noise stripe detection; adaptive/aggressive/frequency-aware suppression modes |
| **3. Fault Classification** | 6-model ensemble (SVM, RF, LDA, MLP, ResNet18, EfficientNet-B0) | PCA dimensionality reduction; stratified K-Fold cross-validation; comprehensive metric suite |

---

## Key Results

### MVDR Beamforming: SINR Improvement

| Metric | Value |
|--------|-------|
| **SINR Improvement** | **+22.66 dB** (strong interference environment) |
| Array geometry | Multi-arm spiral, far-field |
| Validation method | RIR-based realistic acoustic simulation |

### Classification: 100% Accuracy Across All Models

*4-class fault diagnosis on enhanced spectrograms (test set: 54 samples, stratified split)*

| Model | Accuracy | Precision | Recall | F1 | G-Mean | Train Time |
|-------|----------|-----------|--------|----|----|------------|
| **SVM (RBF)** | 100% | 1.00 | 1.00 | 1.00 | 1.00 | 2.5 s |
| **Random Forest** | 100% | 1.00 | 1.00 | 1.00 | 1.00 | 15.7 s |
| **LDA** | 100% | 1.00 | 1.00 | 1.00 | 1.00 | 0.04 s |
| **MLP (Small)** | 100% | 1.00 | 1.00 | 1.00 | 1.00 | 3.9 s |
| **Transfer Learning (ResNet18)** | 100% | 1.00 | 1.00 | 1.00 | 1.00 | 14.5 s |
| **EfficientNet-B0** | 100% | 1.00 | 1.00 | 1.00 | 1.00 | 10.8 s |

### Key Figures

| Figure | Description |
|--------|-------------|
| ![Confusion Matrices](paper/results/confusion_matrices.png) | **Confusion matrices** — all 6 models achieve perfect diagonal (zero misclassification) |
| ![Model Comparison](paper/results/model_comparison.png) | **Model comparison** — accuracy, F1, G-Mean, and training time across all classifiers |
| ![t-SNE Embedding](paper/results/tsne_embedding.png) | **t-SNE visualization** — enhanced features form well-separated, compact clusters per fault type |
| ![Training Curves](paper/results/mlp_loss_accuracy.png) | **MLP training curves** — loss and accuracy convergence |
| ![Transfer Learning Curves](paper/results/transfer_learning_loss_accuracy.png) | **ResNet18 transfer learning curves** |
| ![EfficientNet Curves](paper/results/efficientnet_b0_loss_accuracy.png) | **EfficientNet-B0 3-phase training curves** |

---

## Table of Contents

- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Module 1: MVDR Beamforming](#module-1-mvdr-beamforming)
- [Module 2: Dual-Branch Transformer](#module-2-dual-branch-transformer)
- [Module 3: Fault Classification](#module-3-fault-classification)
- [Paper & Documentation](#paper--documentation)

---

## Project Structure

```text
.
├── mvdr/                      # MVDR beamforming (MATLAB)
│   ├── src/                   #   Core processing modules
│   ├── configs/               #   Experiment configurations
│   ├── data/                  #   Mic positions & audio
│   └── output/                #   Beamforming results
├── transformer/               # Dual-Branch Transformer (Python/PyTorch)
│   ├── src/                   #   Model, training, data pipeline
│   ├── configs/               #   YAML configuration files
│   ├── scripts/               #   train.py, evaluate.py
│   ├── datasets-npz/          #   Input spectrogram data
│   ├── experiments/           #   Checkpoints & logs
│   └── enhancement_report/    #   Evaluation outputs
├── classification/            # Fault classification (Python)
│   ├── data/                  #   Split datasets (train/val/test)
│   ├── results/               #   Model checkpoints & reports
│   ├── split_dataset.py       #   Dataset splitting & K-Fold
│   └── train_classifier.py    #   Multi-model training
├── RIR-Generator/             # Room impulse response generator (C++/MATLAB)
├── paper/                     # LaTeX paper & figures
├── scripts/                   # Utility shell scripts
└── docs/                      # Extended documentation
```

---

## Quick Start

### 1. MVDR Beamforming

```bash
# MATLAB: Run the main MVDR pipeline
# Open MATLAB in mvdr/ directory, then:
mvdr_main
# or for multi-beta parameter sweep:
mvdr_multi_beta_analysis
```

### 2. Transformer Data Generation

```bash
cd transformer
python scripts/train.py --config configs/train_pairwise.yaml
```

### 3. Recollect Figures

```bash
bash scripts/recollect_batch_pairwise.sh
```

### 4. Split Dataset for Classification

```bash
cd classification
python split_dataset.py                          # default: 70/15/15 with 5-fold CV
python split_dataset.py --k-folds 10             # 10-fold CV
python split_dataset.py --no-cv                  # skip K-Fold generation
python split_dataset.py --use-noisy              # use noisy spectrogram instead of enhanced
```

- **Data source**: `transformer/datasets-npz/{DCBias, Harmonic, Loosen, PartialDischarge}/*.npz`
- **Output dir**: `classification/data/`

### 5. Train Classifiers

```bash
cd classification
python train_classifier.py                  # train all models
python train_classifier.py --models svm rf  # train only SVM + Random Forest
python train_classifier.py --use-cv         # use K-Fold CV instead of fixed val split
python train_classifier.py --no-pca         # disable PCA dimensionality reduction
```

**Output**:
```text
classification/results/
├── model_comparison.csv
├── model_comparison.json
├── confusion_matrices.png
├── tsne_embedding.png
├── model_comparison.png
├── <model>_loss_accuracy.png      (MLP / Transfer Learning)
├── svm_report.json
└── ...
```

---

## Module 1: MVDR Beamforming

MVDR (Minimum Variance Distortionless Response) beamforming on a custom **multi-arm spiral microphone array** for directional acoustic signal acquisition.

### Key Features

- Microphone coordinate reading & processing
- Room parameter configuration (RIR generation)
- 3D visualization of room, sources, and microphone array
- Fraunhofer far-field criterion verification
- RIR generation via `RIR-Generator`
- Signal convolution with noise addition
- STFT processing
- Steering vector construction based on actual positions
- Multi-$\beta$ parameter sweep analysis

### Usage

Modify the config name in `mvdr_main.m`:

```matlab
experiment_name = 'real_signal_dcbias_4k';   % or 'demo_2k_4k'
```

Then execute `mvdr_main.m`. For multi-$\beta$ analysis, modify `mvdr_multi_beta_analysis.m` similarly.

### Key Files

| File | Description |
|------|-------------|
| `src/mvdr_main.m` | Main MVDR pipeline |
| `src/mvdr_processing_module.m` | Core MVDR processing |
| `src/rir_generation_module.m` | RIR generation wrapper |
| `src/load_config.m` | Configuration loader |
| `src/load_data_module.m` | Data loading utilities |
| `src/diagnostics_and_visualization_module.m` | Visualization tools |
| `src/mvdr_multi_beta_analysis.m` | Multi-$\beta$ parameter sweep |
| `src/batch_mvdr_raw_datasets.m` | Batch processing for datasets |

---

## Module 2: Dual-Branch Transformer

A dual-branch Transformer model for **speech enhancement**. Inputs are noisy speech spectrum $x_s$ and noise reference spectrum $x_n$; outputs are enhanced speech and time-frequency mask.

> **NOTE**:
> - Training runs on a remote server; results are saved remotely. Use `Remote Host` to check file locations.
> - Errors and package installations must be resolved on the remote server's terminal.
> - `num_frames = (audio_duration × sample_rate - n_fft) // hop_length + 1`. Adjust `input_size` time dimension accordingly (the second parameter of `input_size: (128, 64)`).
> - `num_workers = 0`
> - New datasets renamed locally will NOT sync to remote. Rename on the remote server directly, or add datasets on the remote server.

### Model Architecture (Dual-Branch Transformer)

```mermaid
flowchart TD
    subgraph Speech Branch
        A1[Noisy Speech Mag] --> B1[PatchEmbedding]
        B1 --> C1[TransformerEncoder]
    end
    subgraph Noise Branch
        A2[Noise Reference Mag] --> B2[PatchEmbedding]
        B2 --> C2[TransformerEncoder]
    end
    C1 --> D[CrossAttentionFusion]
    C2 --> D
    D --> E[Decoder]
    E --> F[Mask]
    F --> G["Enhanced = Mask × Noisy Magnitude"]
```

| Component | Description |
|-----------|-------------|
| **PatchEmbedding** | Splits 2D time-frequency map into patches and maps to tokens |
| **TransformerEncoder** | Shared encoder processing speech and noise branches separately |
| **CrossAttentionFusion** | Cross-attention + gating mechanism to suppress noise features |
| **Decoder** | Decodes tokens to time-frequency mask and upsamples to original size |
| **DualBranchTransformer** | Main model; outputs `enhanced = mask * noisy_magnitude` and mask |

### CrossAttentionFusion — 3 Suppression Modes

Three suppression modes are available (default: `frequency_aware`):

| Mode | Mechanism | Use Case |
|------|-----------|----------|
| `adaptive` | Soft gating $z_s - g \cdot z_{cross}$ | Baseline comparison |
| `aggressive` | Dual suppression: amplify interference → subtract + enhance difference | Strong noise |
| `frequency_aware` | Frequency-aware: detect horizontal stripes → enhance recognition → aggressive removal | Remove horizontal noise stripes **(recommended)** |

**Frequency-Aware Mode** adds asymmetric convolution detection:

- **Horizontal detector** (long kernel `kernel_size=5`): Detects frequency-domain continuity (noise stripe features)
- **Vertical detector** (short kernel `kernel_size=3`): Detects vertical speech structure (harmonics/formants)

$$\text{stripe\_score} = \text{concat}(h_{\text{response}}, 1 - v_{\text{response}})$$
$$\text{freq\_attention} = \text{stripe\_enhancer}(\text{stripe\_score})$$

### Training

#### Local Training (Recommended First)

```bash
cd transformer
python scripts/train.py --config configs/train.yaml
```

#### Evaluation (No Re-Training)

```bash
cd transformer
python scripts/evaluate.py --config configs/train.yaml \
    --checkpoint experiments/train/best_model.pth \
    --output-dir enhancement_report --num-samples 20
```

#### Data Generation (Pairwise Mode)

```bash
python transformer/scripts/train.py --config transformer/configs/train_pairwise.yaml
```

#### Docker Training

```bash
docker compose up --build
```

Configuration is specified in `docker-compose.yaml`. The container training script is defined by `Dockerfile` and `docker-compose.yaml`.

**Common checks**:
- Config file exists: `train.yaml`
- Training script exists: `train.py`
- Data directories have content: `data/speech_enhancement/clean` and `data/speech_enhancement/noise`

### Data Processing & Dataset

`SpeechEnhancementDataset` supports two modes:
- Real data directory (`clean/`, `noise/`, optional `noisy/`)
- Auto-generate synthetic samples when no data exists

Provides: audio loading, noisy synthesis (random SNR), Mel spectrogram construction, size alignment, dual-channel features (`mel + delta`).

#### Data Mixing Modes

| Mode | Clean-Noise Relation | Implementation | Use Case | Advantage |
|------|---------------------|----------------|----------|-----------|
| Fixed Pairing | One-to-one | `clean[i] + noise[i]` | Specific room/device | Simulates real environment |
| Random Mixing | No correspondence | `clean[i] + noise[j]` | General noise suppression | High data diversity |

This project uses **random mixing**, which:
- Increases data diversity ($N_{clean} \times N_{noise}$ combinations)
- Trains the model to learn general noise suppression (not memorizing specific pairs)
- Avoids overfitting to specific noise types, improving generalization

#### Dataset Split Strategy

```text
Total Data (100%)
├── Training Set (70%)   — Model learning
├── Validation Set (15%) — Hyperparameter tuning & early stopping
└── Test Set (15%)       — Final evaluation (fully independent)
```

Clean and noise files are split independently:

| Split | Description |
|-------|-------------|
| Clean Set | Independently split into `train_clean / val_clean / test_clean` |
| Noise Set | Independently split into `train_noise / val_noise / test_noise` |

**Advantages**:
- Prevents data leakage: test set clean files never pair with training set noise files
- Handles unequal counts: clean and noise can have different numbers of files
- Ensures fair evaluation: testing only uses `test_clean + test_noise` combinations

#### Data Directory Structure

```text
data/speech_enhancement/
├── clean/                # Clean speech files
│   ├── speech_001.wav
│   ├── speech_002.wav
│   └── ...
├── noise/                # Noise files
│   ├── noise_001.wav
│   ├── noise_002.wav
│   └── ...
└── noisy/                # (Optional) Pre-generated noisy speech
    ├── noisy_001.wav
    └── ...
```

### Training & Validation

- **`compute_loss`**: Composite loss (enhanced MSE + mask reconstruction MSE + sparse regularization)
- **`train_one_epoch`**: Standard training loop (backpropagation, gradient clipping, logging)
- **`validate`**: Validation set loss evaluation
- **`main`**: Complete training pipeline (config, DataLoader, optimizer, scheduler, best/checkpoint saving)

### Evaluation & Visualization

- **`evaluate_speech_quality`**: Statistics (MSE, SNR before/after, improvement; PESQ/STOI interfaces reserved but not currently computed)
- **`save_enhanced_audio`**: Saves enhanced results as `.npz` spectrogram data
- **`visualize_spectrogram_comparison`**: Generates 4-panel comparison (clean/noisy/enhanced/mask)
- **`generate_enhancement_report`**: Outputs image + text report

### Evaluation Metrics

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| SNR Improvement | Signal-to-noise ratio improvement | dB | ↑ Higher |
| MSE | Spectrogram mean squared error | $[0, \infty)$ | ↓ Lower |
| PESQ* | Perceptual speech quality | $[1, 4.5]$ | ↑ Higher |
| STOI* | Short-time objective intelligibility | $[0, 1]$ | ↑ Higher |

*\*Interface reserved, not currently computed.*

### Visual Quality Assessment

| Comparison | Good Performance | Poor Performance |
|------------|-----------------|------------------|
| Noisy vs Clean | — | Large red noise regions |
| Enhanced vs Clean | Similar spectrogram structure | Over-smoothed / distorted |
| Mask Distribution | Noise region ≈ 0, speech region ≈ 1 | All 0 or all 1 |
| SNR Improvement | +5 dB or more | < 2 dB or negative |

**Color guide**:
- Spectrogram (`inferno` colormap): Red = high energy, Black = low energy
- Mask (`viridis` colormap): Yellow = retain (≈1), Purple = suppress (≈0)

**Visual inspection**:
- **Noisy spectrogram**: Visible horizontal stripes (periodic noise) or random spots (broadband noise)
- **Enhanced spectrogram**: Noise texture disappears; speech formants (horizontal stripes) become clearer
- **Mask**: Noise regions appear deep purple (suppressed); speech regions appear yellow (retained)

### Inference & Utilities

- **`test_inference`**: Random input shape/value range check
- **`prepare_dataset_structure`**: Generates data directory template
- **`generate_and_save_noisy_audio`**: Batch generates noisy audio and writes to `noisy/`

### Environment

- Framework: PyTorch
- Dependencies: `numpy`, `matplotlib`, `librosa`, etc.
- Docker containerization supported

---

## Module 3: Fault Classification

Multi-model classification on enhanced spectrograms for power equipment fault diagnosis.

### Dataset Classes

| Class | Label | Description |
|-------|-------|-------------|
| DC Bias | 0 | DC bias fault |
| Harmonic | 1 | Harmonic distortion |
| Loosen | 2 | Mechanical looseness |
| Partial Discharge | 3 | Partial discharge |

### Models (Ordered by Recommendation for Small Datasets)

| # | Model | Type | Strengths |
|---|-------|------|-----------|
| 1 | **SVM** (RBF kernel) | Traditional ML | Best for small, structured data |
| 2 | **Random Forest** | Ensemble | Robust, handles noise well |
| 3 | **LDA** | Linear | Great if classes are linearly separable |
| 4 | **MLP** (small) | Neural Network | Strong regularization |
| 5 | **Transfer Learning** (CNN) | Deep Learning | Treats spectrograms as images |
| 6 | **EfficientNet-B0** | Deep Learning | Pre-trained CNN for image classification |

### Metrics

| Metric | Description |
|--------|-------------|
| Top-1 Accuracy | Overall classification accuracy |
| Precision | Class-wise precision |
| Recall | Class-wise recall |
| F1-Score | Harmonic mean of precision & recall |
| G-Mean | Geometric mean of class-wise recall |
| ROC-AUC (OVR) | One-vs-rest ROC AUC |
| Confusion Matrix | Per-class prediction breakdown |
| Parameter Count | Model size (PyTorch models) |
| FLOPs | Computational cost (PyTorch models) |

---

## Paper & Documentation

- **Paper**: `paper/mvdr.tex` (LaTeX, IEEE format)
- **Bibliography**: `paper/MVDR.bib`
- **Figure standards**: `docs/IEEE_COLOR_STANDARD.md`, `docs/IEEE_FIGURE_DESCRIPTIONS.md`
- **Multi-$\beta$ analysis**: `docs/MULTI_BETA_ANALYSIS.md`, `docs/README_MULTI_BETA.md`
- **MVDR theory**: `docs/mvdr_of_real_signal_based_on_rir.md`
- **Project structure**: `docs/PROJECT_STRUCTURE.md`
- **Figure caption reference**: `docs/FIGURE_CAPTION_QUICK_REF.md`