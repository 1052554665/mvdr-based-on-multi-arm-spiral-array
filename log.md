> Update the README according the following statement and add some experimental results and figures for presentation, also remove the power transformer background.


The pipeline of this research are as follows:

- The target source A and source B are processed by wideband MVDR, separately, to obtain the enhanced source A but with residual interference from source B, which can be called A1, and the enhanced source B but with residual interference from source A, which can be called B1.
- The enhanced sources are then feeded into a deep neural network (DNN) for further enhancement, which aims to suppress the residual interference and improve the overall quality of the sources, finally producing the final enhanced target sources A.
- The activate function is designed to achieve A1 minus B1.

### Changes Summary

**1. Pipeline Rewrite (A₁ − B₁)**
- Title changed from "Acoustic Fault Diagnosis" → "Acoustic Source Separation"
- Pipeline now clearly describes: Source A & B → separate MVDR → A₁ (enhanced A + residual B), B₁ (enhanced B + residual A) → Dual-Branch Transformer → **A₁ − B₁ activation** → Final enhanced A
- Added two new mermaid diagrams illustrating the full pipeline
- Added "Core Insight: A₁ − B₁ Activation" section explaining the mathematical intuition

**2. Power Transformer Background Removed**
- Removed all "power equipment", "industrial environments" references
- Problem statement now focuses on multi-source acoustic separation in reverberant environments
- Classification module description updated to "acoustic fault diagnosis"

**3. Experimental Results Added**
- **Transformer Enhancement SNR table**: per-class SNR before/after with Δ SNR (DC Bias: +1.54 dB, Harmonic: +1.46 dB, Loosen: +1.28 dB, Partial Discharge: +1.80 dB)
- **MVDR Beamforming Figures section**: 9 figure references (beampattern, PSD, eigenspectra, β sweep, spectrograms)
- **Transformer Enhancement Spectrograms**: location of per-class comparison PNGs
- **System Setup section**: experiment configuration table, equipment list, system diagrams

**4. Module Updates**
- Module 1 (MVDR): Added pipeline description showing separate steering for A and B
- Module 2 (Transformer): Complete rewrite with A₁ − B₁ activation function math, "Why This Works" table, updated mermaid diagram
- Updated component descriptions table for the new architecture
- Updated project structure to include results, output, drawing

>do not present metrics since the results are preliminary and may not reflect the final performance. you can display figure 1 to 11 in `mvdr.tex`as placeholders for the presentation, but please note that these figures are for illustrative purposes only and do not represent final results.

>the figures are same with the ones in `mvdr.tex`