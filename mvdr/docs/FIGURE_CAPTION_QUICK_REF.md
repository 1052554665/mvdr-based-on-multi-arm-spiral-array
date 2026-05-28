# Fig. 2 & Fig. 3 - IEEE论文快速参考卡

## 图注（Figure Captions）- 直接复制粘贴

### Fig. 2
```
Covariance eigenspectra of the full-band received signal 
across wall reflection coefficients β ∈ {0, 0.2, 0.4, 0.6, 0.8}. 
Each curve represents the eigenvalue distribution from a 4-element 
microphone array in a 5×4×6 m³ simulated room. Eigenvalues regularized 
with diagonal loading (ε = 10⁻⁴ Tr(R_xx)/M). The progressive eigenvalue 
spreading illustrates the deteriorating signal subspace conditioning 
in reverberant environments.
```

### Fig. 3
```
MVDR beamformer performance metrics versus wall reflection coefficient: 
(a) SNR gain in target-steered processing (blue circles), showing 18 dB 
improvement in anechoic to 7 dB at β=0.8; (b) ISR gain in interference-steered 
branch (red squares), demonstrating variable suppression from 8–23 dB; 
(c) Condition number κ(R_xx) on logarithmic scale (purple triangles), 
revealing exponential ill-conditioning with reverberation intensity. 
Metrics computed from 16-second speech signals at 16 kHz sampling rate.
```

---

## 论文中的引用段落 - 复制即用

### 引入图表（Introduction）
```
To characterize the MVDR beamformer's performance across 
acoustic conditions, we systematically analyze eigenvalue 
distributions and performance metrics (Fig. 2 and Fig. 3) 
as functions of the room wall reflection coefficient β.
```

### 结果陈述（Results）
```
The eigenvalue analysis presented in Fig. 2 demonstrates 
critical dependence on room acoustics. In anechoic conditions (β=0), 
a distinct eigenvalue separation enables robust subspace identification. 
However, as β increases to 0.8, eigenvalue spreading (Fig. 2) 
correlates with SNR gain reduction from 18 dB to 7 dB (Fig. 3a).
```

### 分析讨论（Analysis）
```
The condition number evolution (Fig. 3c) explains the performance 
limitation through numerical conditioning. The 860× increase in κ(R_xx) 
from β=0 to β=0.8 necessitates regularization strength adjustment from 
ε=10⁻⁴ to ε=10⁻², balancing between stability and adaptivity.
```

### 结论（Conclusion）
```
This comprehensive analysis (Figs. 2–3) reveals that MVDR performance 
in reverberant environments is fundamentally limited by eigenvalue 
degradation and covariance ill-conditioning. The results underscore 
the necessity of adaptive regularization strategies for practical deployment.
```

---

## 一句话总结 - 不同场景

| 论文部分 | 推荐表达 |
|---------|---------|
| **摘要** | "MVDR performance degrades from 18 dB SNR gain (anechoic) to 7 dB (β=0.8 reverberant)" |
| **引言** | "We evaluate MVDR robustness across wall reflection coefficients using eigenvalue and performance analysis (Figs. 2–3)" |
| **方法** | "Covariance eigenspectra and SINR metrics are analyzed as functions of β" |
| **结果** | "Fig. 2 shows progressive eigenvalue spreading; Fig. 3 quantifies resulting performance degradation" |
| **讨论** | "The eigenvalue-performance correlation (Figs. 2–3) suggests ill-conditioning is the primary limiter" |
| **结论** | "Adaptive regularization is essential for MVDR deployment in reverberant rooms, as demonstrated by Fig. 3(c)" |

---



## 常见问题 - 论文评审应对

**Q: "为什么选择这5个β值？"**  
A: "The parameter range β ∈ {0, 0.2, 0.4, 0.6, 0.8} covers practical scenarios from anechoic test environments to untreated classrooms. This spacing provides sufficient resolution to observe the eigenvalue degradation trend and its performance consequences."

**Q: "为什么特征值在β=0.8时仍未完全混合？"**  
A: "Residual eigenvalue separation at β=0.8 is due to the sustained coherence of direct and early reflection paths from the two point sources, which maintain distinct spatial signatures despite late reverberation diffuseness."

**Q: "ISR增益的非单调性说明了什么？"**  
A: "The non-monotonic behavior in Fig. 3(b) reflects the complex interplay between the interference's multipath structure and its spatial decorrelation due to reverberation, occasionally enhancing interference suppression at intermediate β values."

**Q: "这些结果能推广到其他阵列吗？"**  
A: "The eigenvalue degradation pattern (Fig. 2) is qualitatively universal for all microphone arrays in reverberant spaces. Quantitative metrics (Fig. 3) depend on array geometry and room dimensions, but the fundamental trends remain consistent."

---


