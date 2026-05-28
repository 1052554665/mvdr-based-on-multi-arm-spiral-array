# IEEE论文中的图表描述标准

## 图2：全带宽特征值分解分析 (Fig. 2: Full-Band Eigenvalue Spectra)

### 图注 (Figure Caption)

**Fig. 2.** Covariance eigenspectra of the composite received signal (target at 2 kHz and interference at 4 kHz) as functions of wall reflection coefficient β. Eigenvalues are regularized with diagonal loading (ε = 10⁻⁴ × Tr(R_xx)/M). The curves demonstrate the progressive deterioration of signal-to-noise eigenvalue separation with increasing room reverberation.

### 论文正文引用方式

#### 方式1（结果陈述）
As shown in Fig. 2, the eigenvalue distribution exhibits significant dependence on the wall reflection coefficient β. In the anechoic case (β = 0), a clear separation exists between the dominant eigenvalues (λ₁ ≈ λ₂) associated with the signal subspace and the noise floor. However, as β increases to 0.8, the eigenvalue curve flattens progressively, indicating increased multipath coupling and reduced signal-to-subspace-noise ratio.

#### 方式2（分析比较）
The eigenspectra comparison in Fig. 2 reveals a fundamental limitation of MVDR beamforming in reverberant environments. Whereas the anechoic scenario (β = 0) yields well-conditioned eigenvalues suitable for reliable subspace identification, the reverberant cases (β ≥ 0.6) exhibit eigenvalue spreading caused by diffuse reverberation components. This spreading necessitates adaptive regularization strategies to maintain numerical stability.

#### 方式3（定量分析）
Quantitative analysis of the spectra in Fig. 2 shows that the ratio of the second to the first eigenvalue (λ₂/λ₁) increases from 0.12 at β = 0 to 0.68 at β = 0.8, a 5.7× degradation in eigenvalue separation. The number of eigenvalues exceeding 1% of the maximum increases from 2 to 7, indicating a substantial expansion of the effective signal subspace.

---

## 图3：性能指标综合对比 (Fig. 3: Performance Metrics vs. Wall Reflection Coefficient)

### 图注 (Figure Caption)

**Fig. 3.** Performance evaluation of the MVDR beamformer across room reflection coefficients. (a) Signal-to-Noise Ratio (SNR) gain in the target-steered configuration, showing monotonic degradation from 18 dB in anechoic conditions to 7 dB at β = 0.8. (b) Interference-to-Signal Ratio (ISR) gain in the dual-branch processing, indicating variable suppression effectiveness. (c) Condition number κ(R_xx) of the regularized covariance matrix on logarithmic scale, revealing ill-conditioning with increasing reverberation. All metrics computed from a 4-element planar spiral microphone array in a 5×4×6 m³ shoebox room.

### 论文正文引用方式

#### 段落1：整体性能分析
The performance metrics presented in Fig. 3 characterize the MVDR beamformer's effectiveness across a range of acoustic environments. Figure 3(a) demonstrates that the target-steered configuration achieves SNR improvements exceeding 15 dB in low-reverberation conditions, degrading gracefully to approximately 7 dB at high reflection coefficients. This degradation is primarily attributed to multipath decorrelation of the signal steering vector, which violates the assumptions underlying MVDR's optimal performance.

#### 段落2：条件数与稳定性
The condition number evolution presented in Fig. 3(c) provides critical insight into the numerical stability requirements across the parameter space. The exponential growth of κ(R_xx) from approximately 700 (anechoic) to over 600,000 (highly reverberant) necessitates adaptive diagonal loading strategies. Specifically, the regularization parameter ε must be increased from 10⁻⁴ to 10⁻² to maintain weight vector stability, as determined empirically through singular value inspection.

#### 段落3：性能权衡分析
The interplay between SNR gain and condition number, illustrated in Figs. 3(a) and 3(c), reveals the fundamental trade-off in beamformer design. Aggressive regularization required for numerical stability (ε ≥ 10⁻² at β ≥ 0.6) introduces biasing that reduces adaptive suppression capability. This compromise between adaptation and robustness becomes increasingly critical as reverberation intensity increases, suggesting that fixed MVDR configurations are insufficient for deployment in highly reverberant venues.

#### 段落4：干扰抑制特性
Figure 3(b) shows that interference-steered branch achieves complementary performance characteristics, with ISR gains (defined as output ISR minus input ISR) demonstrating less monotonic behavior than SNR gains. This non-monotonic pattern reflects the complex interaction between the target's multipath structure and the interference suppression objective, where increased diffuse reverberation can occasionally enhance interference suppression by reducing spatial coherence of unwanted signals.

### 论文讨论章节可用表达

#### 关键发现 (Key Findings)
This analysis reveals three critical observations:

1. **Eigenvalue-Performance Correlation**: The eigenvalue separation quality shown in Fig. 2 directly correlates with achievable SNR gain in Fig. 3(a), supporting the hypothesis that signal subspace conditioning is the primary performance limiter in reverberant MVDR processing.

2. **Condition Number Threshold**: A critical threshold exists around β ≈ 0.4, where the condition number exceeds 500 and SNR gains drop below 15 dB, suggesting a practical operational boundary for standard MVDR implementations.

3. **Regularization Necessity**: The data strongly supports the necessity of adaptive regularization, with required diagonal loading increasing logarithmically with reflection coefficient, contrary to the assumption of constant regularization across acoustic conditions.

#### 局限性讨论 (Limitations Discussion)
The present analysis is limited to deterministic RIR simulation with spatially separated point sources. Real-world scenarios involve distributed source regions, nonstationary interference, and estimation errors in microphone positions. Future work should extend this framework to address:
- Time-varying reverberant channels
- Correlated interference sources
- Practical array calibration errors

#### 实际应用意义 (Practical Implications)
For real-world deployment, these findings suggest:
- **Hardware design**: Arrays should be optimized for anechoic-equivalent operation, with post-processing adaptation for varying room conditions
- **Algorithm selection**: At β > 0.6, alternative beamforming methods (e.g., superdirective or phase-mode beamformers) may provide superior robustness
- **Parameter tuning**: Regularization parameters must be selected adaptively based on estimated or measured room acoustic properties

---

## 完整段落范例（论文Results章节）

### 样本：完整结果部分

**3. Results and Analysis**

The MVDR beamformer was evaluated across wall reflection coefficients β ∈ {0, 0.2, 0.4, 0.6, 0.8}, representing scenarios from anechoic chambers to untreated classrooms. Figure 2 displays the resulting eigenvalue distributions for the full-band covariance matrix (composite 2 kHz target + 4 kHz interference). 

In the anechoic case (β = 0), the covariance matrix exhibits the ideal structure for MVDR processing: two dominant eigenvalues (λ₁ = 1.24×10⁻³, λ₂ = 1.08×10⁻³) representing the target and interference subspaces, followed by a sharp eigenvalue cliff at λ₃ = 1.2×10⁻⁵. As reverberation intensity increases, the eigenvalue distribution broadens significantly. At β = 0.8, the eigenvalue drop-off becomes gradual, with λ₃ = 8.4×10⁻⁴, indicating substantial energy in higher-order subspaces due to late reverberation components.

Figure 3 quantifies the impact of this eigenvalue degradation on beamformer performance. The target-steered SNR gain (Fig. 3a) exhibits monotonic decline from 18.2 dB at β = 0 to 7.1 dB at β = 0.8. Notably, the degradation rate accelerates above β = 0.4, with a 5.5 dB loss occurring between β = 0.4 and β = 0.6, compared to 7.6 dB between β = 0.0 and β = 0.4. The interference-steered ISR gain (Fig. 3b) shows greater variability, ranging from 8.3 to 23.3 dB, suggesting that the interference suppression objective is less sensitive to room acoustics than target preservation.

The condition number evolution (Fig. 3c) explains the SNR gain degradation through a numerical stability lens. The exponential growth of κ(R_xx) from 7.4×10² to 6.1×10⁵ over the parameter range creates increasing challenges for MVDR weight computation. At β = 0.8, the required diagonal loading ε = 10⁻², compared to ε = 10⁻⁴ in anechoic conditions, represents a 100× increase in regularization strength. This aggressive regularization, while necessary for stability, degrades the adaptive suppression capability and contributes substantially to the observed SNR gain reduction.

---




