>merge the content from `transformer/README.md` into the root `README.md` and update it if necessary.


**Merged README.md → root README.md** with the following structure:

| Section | Content |
|---------|---------|
| **Title & Pipeline** | New project title + Mermaid flowchart showing the full pipeline |
| **Project Structure** | Unified directory tree covering all 3 modules |
| **Quick Start** | All original commands preserved (data generation, recollect, split, train) |
| **Module 1: MVDR** | Key features, usage, and file descriptions for the MATLAB beamforming module |
| **Module 2: Transformer** | Full content from README.md — architecture, 3 suppression modes, training, data processing, evaluation metrics, visual quality guide |
| **Module 3: Classification** | Expanded with dataset classes table, model recommendation priority, and metrics definitions |
| **Paper & Docs** | Cross-references to all documentation files |

Key improvements:
- Added a **pipeline diagram** showing the end-to-end workflow
- Added a **Table of Contents** for navigation
- Preserved all original commands and notes from both files
- Added the **fault class labels table** and **model comparison table** for the classification module
- The original README.md is left intact as module-level documentation

Made changes.

> also state the problem, method, key result (with figures).

| Section | Content |
|---------|---------|
| **Problem** | Two fundamental challenges: spatial interference (poor SINR in industrial environments) and spectrogram-level residual noise that degrades classification |
| **Method** | 3-stage pipeline diagram (Mermaid) + innovation table — MVDR spatial filtering → Dual-Branch Transformer enhancement → 6-model classification |
| **Key Results** | SINR improvement **+22.66 dB** from MVDR; **100% classification accuracy** across all 6 models (54-sample test set, stratified); 6 key figures referenced with descriptions (confusion matrices, model comparison, t-SNE, training curves) |

All figures are linked from results (confirmed existing in the repo) and the results table is populated from actual model_comparison.csv data.

Made changes.