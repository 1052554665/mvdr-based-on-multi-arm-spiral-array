# Dataset Division & Model Selection

>There are four class, each has ninty samples. How to divide the dataset properly? Is it suitable for 70:15:15? 
which models are suitable for classification with such small samples? please list some models from traditional to modern.

## Your Dataset at a Glance
- **4 classes × 90 samples = 360 total samples** — this is a **small dataset**


## 70 : 15 : 15 Split — Is It Suitable?

**Yes, 70:15:15 is reasonable**, but let's see what it gives you:

| Split | Ratio | Total Samples | Per Class |
|---|---|---|---|
| Train | 70% | **252** | ~63 |
| Validation | 15% | **54** | ~13 |
| Test | 15% | **54** | ~13 |

✅ The split is **balanced** (equal class sizes), so simple random splitting works well.
✅ 63 samples per class for training is workable for traditional ML models.
⚠️ It's **tight for deep learning** — DNNs typically need thousands of samples per class.

> **Important:** Use **stratified splitting** to ensure each split maintains the 4-class proportion.

```python
from sklearn.model_selection import train_test_split

# Stratified split: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)
```

## Consider: Stratified K-Fold Cross-Validation

With only ~63 training samples per class, **K-Fold CV (k=5 or k=10)** is often a better strategy than a fixed val split — it makes fuller use of your limited data.

```
K-Fold on Train Set (252 samples):
  Fold 1: [Val] [Tr] [Tr] [Tr] [Tr]
  Fold 2: [Tr] [Val] [Tr] [Tr] [Tr]
  ...
  → Average performance = more reliable validation signal
```


## Models: Traditional → Modern

Given your small dataset (~360 samples, 4 classes), here is a recommended progression:

### 🟢 Traditional ML — Best Starting Point

| Model | Why Suitable |
|---|---|
| **Logistic Regression** | Simple, interpretable, strong baseline for small data |
| **Linear/Quadratic Discriminant Analysis (LDA/QDA)** | Excellent for small, balanced multi-class problems |
| **Support Vector Machine (SVM)** | Extremely effective with small samples; kernel trick handles non-linearity |
| **K-Nearest Neighbors (KNN)** | No training needed; works well when classes are separable |
| **Naive Bayes** | Very fast; good when features are (roughly) independent |
| **Decision Tree** | Interpretable; prone to overfit — use with pruning |

### 🟡 Ensemble Methods — Strong Performers

| Model | Why Suitable |
|---|---|
| **Random Forest** | Reduces overfitting via bagging; robust on small data |
| **Gradient Boosting (XGBoost, LightGBM)** | High accuracy; handles small tabular datasets very well |
| **AdaBoost** | Combines weak learners; effective with limited data |

### 🔵 Modern / Deep Learning — Use with Caution

| Model | Notes |
|---|---|
| **MLP (Small, shallow)** | 1–2 hidden layers + heavy dropout; feasible but risky |
| **Transfer Learning (CNN)** | ✅ If data is images — freeze pretrained backbone (ResNet, EfficientNet), fine-tune only the head |
| **Pretrained Transformers** | ✅ If data is text — fine-tune BERT/DistilBERT on your classes |
| **Prototypical Networks** | Few-shot learning approach; designed for small-sample classification |

---

## Recommendation Priority for Your Case

```
1st  →  SVM (RBF kernel)          ← Often best for small, structured data
2nd  →  Random Forest / XGBoost   ← Robust, handles noise well  
3rd  →  LDA                       ← Great if classes are linearly separable
4th  →  Transfer Learning          ← Only if your data is images or text
❌   →  Training deep nets from scratch  ← Avoid with 360 samples
```

> **Golden rule for small datasets:** Prefer models with **fewer parameters and stronger inductive biases**. Save deep learning for when you have data augmentation, transfer learning, or thousands more samples.

# The dual-channel transformer only does signal enhancement. It has zero classification capability

>check the transformer project, is the model only can be used to enhance the signal? dose it have the function to image classification? 

The model is purely a speech enhancement model — no classification capability.
What it does (and only does): speech denoising
The DualBranchTransformer (dual_branch_transformer.py:131) is a two-input, one-output denoising model.

# Dataset Splitting & Model Implementation

Creating a script for splitting the dataset and implementing some neural networks for image classification in project `classification`. The processes are as follows:

- Create a script to split the dataset into training, validation, and testing sets with a specified ratio: 70% training, 15% validation, 15% testing. The directory is located at `transformer/datasets-npz/`, and the spilted datasets should be saved in `classification` project. Here are some tips to split the dataset:
  - Use **stratified splitting** to ensure each split maintains the 4-class proportion.
      ```python
      from sklearn.model_selection import train_test_split

      # Stratified split: 70% train, 15% val, 15% test
      X_train, X_temp, y_train, y_temp = train_test_split(
          X, y, test_size=0.30, stratify=y, random_state=42
      )
      X_val, X_test, y_val, y_test = train_test_split(
          X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
      )
      ```

  - Stratified K-Fold Cross-Validation
    With only ~63 training samples per class, **K-Fold CV (k=5 or k=10)** is often a better strategy than a fixed val split — it makes fuller use of limited data.

    ```
    K-Fold on Train Set (252 samples):
        Fold 1: [Val] [Tr] [Tr] [Tr] [Tr]
        Fold 2: [Tr] [Val] [Tr] [Tr] [Tr]
        ...
        → Average performance = more reliable validation signal
    ```

- The recommendation priority of networks are as follows:
    ```
    1st  →  SVM (RBF kernel)          ← Often best for small, structured data
    2nd  →  Random Forest / XGBoost   ← Robust, handles noise well  
    3rd  →  XGBoost          ← Great for tabular data
    4th  →  LDA                       ← Great if classes are linearly separable
    5th  →  MLP (small)      ← With strong regularization
    6th  →  Transfer Learning          ← Only if your data is images or text
    ```
