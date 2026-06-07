> Why the SVM, RF, and XGBoost t-SNEs show the input space while the others (LDA, MLP, CNNs) show learned embeddings?

---

This gets at a fundamental distinction in how these models work internally.

## The Core Reason: Does the model learn an internal representation?

**SVM, RF, and XGBoost** are "shallow" models that **don't create intermediate learned representations** — they operate directly on the input feature space. So when you apply t-SNE to visualize what they "see," you're visualizing the *original input features*, because that's the only space these models ever work in.

**LDA, MLP, and CNNs** explicitly construct a **transformed internal space** as part of their mechanism. t-SNE on these models visualizes *that learned space*, not the input.

---

## Model-by-Model Breakdown

### Input Space Models (SVM, RF, XGBoost)
| Model | Why it's the input space |
|---|---|
| **SVM** | Finds a hyperplane in the input space (or kernel-mapped space, but no intermediate layer stores activations). There's no "hidden representation" to extract. |
| **RF** | Each tree splits on raw input features. The ensemble vote has no latent embedding — it's a direct function of inputs. |
| **XGBoost** | Same logic: boosted trees operate on input features iteratively. No learned dense embedding exists. |

For these, t-SNE is applied to **X** (your raw features), which is why plots from SVM/RF/XGBoost look identical to each other — they're all showing the same input data.

---

### Learned Embedding Models (LDA, MLP, CNNs)

| Model | What embedding is extracted |
|---|---|
| **LDA** | Projects data onto axes that **maximize class separability** — a linear transform of inputs into a lower-dim discriminative space. |
| **MLP** | The activations of the **last hidden layer** form a learned representation shaped by backprop to be useful for classification. |
| **CNN** | The **flattened output of the final conv/pooling layer** (before the classifier head) is a rich, hierarchically learned embedding. |

These models transform the input into a new geometry. t-SNE on these embeddings shows **how the model has reorganized the data** — which is why CNN embeddings typically show much tighter, better-separated clusters than the raw input.

---

## Why This Matters Visually

```
Input space t-SNE          Learned embedding t-SNE
(SVM/RF/XGBoost)           (MLP/CNN)

  ·  ·× ·  ○              ···    ×××
 · ×·  ○ ○                  ···  ×××
  × · ○  ·○                      ○○○○
 ·  × ○○ ·                     ○○○○

(messy, overlapping)       (cleaner, clustered)
```

If your CNN t-SNE shows tighter clusters than your SVM t-SNE, that's the model's learned representation actively **doing work** — separating classes in a way the raw input space couldn't.

---

## The Practical Takeaway

> **You can only extract a t-SNE embedding from a layer that exists.** Tree-based models and SVMs have no hidden layers, so the only thing to visualize is the input. Representation-learning models (LDA, MLP, CNN) give you a meaningful alternative space to inspect.

This is also why embedding visualization is much more *informative* for deep models — you're seeing what the network *learned to care about*, not just what you handed it.