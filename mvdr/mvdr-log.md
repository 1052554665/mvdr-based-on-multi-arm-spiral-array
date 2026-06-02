# MVDR Algorithm and Waveform Duration

>The target and interference signal are 1 second waveform, is it correct for mvdr algorithm?

Based on my analysis of your MVDR implementation, **yes, a 1-second waveform is generally appropriate**, but here's the technical breakdown:

## Key Points About Waveform Duration for MVDR:

**✓ What's Generally Correct:**
- **1 second is a reasonable duration** for MVDR beamforming analysis, especially for:
  - **Covariance matrix estimation**: Your code uses `Mavg = 31` frames (averaging window), which needs sufficient data to compute stable covariance matrices
  - **Statistical stability**: At 16kHz sampling, 1 second = 16,000 samples, providing adequate statistics for accurate spatial filtering

**Calculation:**
- 1 second @ 16kHz = 16,000 samples
- With window length = 512, step = 256: ~62 frames of data
- Your averaging window (31 frames) can slide ~31 times across the signal
- This gives reasonable covariance estimates

**⚠️ Considerations:**

1. **Minimum duration** depends on:
   - Array geometry and number of microphones
   - STFT parameters (your current config: 1024 FFT, 512 window)
   - Covariance averaging window (31 frames)
   - For 32+ microphones: ideally 2-5 seconds for stable estimates

2. **Signal stationarity**: 
   - MVDR works best with quasi-stationary signals (1 second is fine for this)
   - If target/interference change characteristics, longer is better

3. **Frequency resolution impact**:
   - At 16kHz fs with 1024 FFT: ~15.6 Hz per bin
   - For your frequency range (300Hz-8kHz): sufficient resolution

**Recommendations:**
- ✓ **Keep 1 second** if your signals are stationary
- 📈 **Increase to 3-5 seconds** if you're seeing unstable beamformer weights
- Check covariance matrix conditioning: if eigenvalue ratio is > 1000, increase duration

Would you like me to analyze your covariance matrix stability or add diagnostic plots to verify if 1 second is sufficient?



# Research Plan Analysis and Adjustments

>A research plan are as follows: 
- There is a divided dateset @raw_datasets, the ratio of the train, val and test is 6:2:2, every subset include ten classes, normal and other nine fault.
- In each subset, the normal class is defined as interference, and other fault classes are target signal, separatly. Then implementing mvdr algorithm, a spectrum dataset is acquired. Is it correct?
- The spectrum dataset is the input of the @file:dual_branch_transformer.py model. Finally, A enhanced spectrum dataset is get, which is processed by dual-channel Transformer network.
- The enhanced spectrum dataset is as the input of classification network.he above plan

## ✓ **What's Correct:**

1. **Dataset split (6:2:2)** - ✓ Standard practice
2. **Dual-branch MVDR concept** - ✓ Treating normal as "interference" and faults as "target" for beamforming makes sense for fault isolation
3. **Pipeline flow** - ✓ The general MVDR → Transformer → Classification sequence is reasonable

---

## ⚠️ **Critical Issues to Address:**

### **Issue 1: Dual-Branch Transformer Input Dimension Mismatch**
Looking at `dual_branch_transformer.py`, the model expects:
```python
in_channels: int = 2  # Two channels!
input_size: tuple[int, int] = (128, 64)  # freq x time
```

**Problem**: Your MVDR algorithm outputs **individual spectra** (magnitude/phase) per branch, but the Transformer expects **2-channel input** (e.g., complex real/imaginary or target/interference interference channels).

**Solution**: Each MVDR output should contain:
- **Channel 1**: Target-steered beamformer output
- **Channel 2**: Interference-steered beamformer output

### **Issue 2: MVDR Class-wise Application Strategy**
Your plan says "normal class → interference, fault classes → target signal, **separately**." This needs clarification:

**Current interpretation** (⚠️ problematic):
```
For each fault class:
  - Mix all normal samples as interference
  - Mix current fault class as target
  - Generate MVDR spectra
→ Creates redundant processing
```

**Better approach**:
```
Option A (Recommended for fault detection):
  - All non-fault = interference reference
  - Each fault class = target
  - Single MVDR run per sample pair
  
Option B (Multi-class MVDR):
  - Create MVDR outputs for all class pairs
  - Expensive but captures inter-class distinctions
```

### **Issue 3: Oracle Covariance Configuration**
Your MVDR config has:
```matlab
config.use_oracle_intnoi_cov = true;   % Use true interference+noise cov
config.use_oracle_tarnoi_cov = true;   % Use true target+noise cov
```

**Problem**: This assumes you **know** the true interference/target signals, which you do in training but **NOT in test time**. This will cause data leakage!

**Solution**:
```matlab
% Training: false (estimate from data)
config.use_oracle_intnoi_cov = false;
config.use_oracle_tarnoi_cov = false;

% Or use adaptive covariance with proper train/val/test splitting
```

### **Issue 4: Spectrum Size Mismatch**
MVDR generates spectrograms with shape:
```
(num_frequencies, num_time_frames)
```

Your Transformer expects `(128, 64)`. You need to **document**:
- STFT window size
- Frequency bins generated
- Resampling strategy to match `(128, 64)`

---

## 📋 **Recommended Revised Plan:**

```
1. DATA PREPARATION
   ├─ Split: train/val/test (6:2:2) ✓
   ├─ Train MVDR params on training data only
   └─ NO oracle covariance in production

2. MVDR PROCESSING (per sample)
   ├─ Input: [target_waveform, interference_waveform]
   ├─ Generate: [target-steered spectrum, interference-steered spectrum]
   ├─ Output shape: (128, 64) → resize if needed
   └─ Stack to 2-channel: shape=(2, 128, 64) ✓

3. DUAL-BRANCH TRANSFORMER
   ├─ Input: 2-channel spectrum
   ├─ Output: Enhanced spectrum + attention mask
   └─ Intermediate features for classification

4. CLASSIFICATION HEAD
   ├─ Input: Enhanced spectrum + optional features
   ├─ Output: 10-class prediction
   └─ Training only on train split
```

---

## 🔧 **Implementation Checklist:**

- [ ] Disable oracle covariance before cross-validation
- [ ] Verify MVDR output shape matches Transformer input `(2, 128, 64)`
- [ ] Clarify MVDR generation strategy for each class pair
- [ ] Add data augmentation to handle fault variations
- [ ] Test pipeline on a small subset first (10-20 samples)
- [ ] Log which config/dataset is used (prevent data leakage)


## 
- For oracle covariance, you can use it during training to analyze upper-bound performance, but **never in validation/test**. Instead, use sample-based covariance estimation or a separate noise-only dataset for interference covariance. This will ensure your model learns to generalize without relying on unrealistic assumptions.

```bash
base_config = load_config(experiment_name);
config.use_oracle_intnoi_cov = false;
config.use_oracle_tarnoi_cov = false;
```

- for MVDR Class-wise Application Strategy, it solved in  `batch_mvdr_raw_dataset.m`.


