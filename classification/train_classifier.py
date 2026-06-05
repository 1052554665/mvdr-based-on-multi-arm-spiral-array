#!/usr/bin/env python3
"""
Train and evaluate multiple classifiers on the split spectrogram dataset.

Models are trained in order of recommendation priority for small datasets:

    1. SVM (RBF kernel)        — often best for small, structured data
    2. Random Forest            — robust, handles noise well
    3. XGBoost                  — great for tabular data
    4. LDA                      — great if classes are linearly separable
    5. MLP (small)              — with strong regularization
    6. Transfer Learning (ResNet18)   — pretrained CNN, frozen vs fine-tuned
    7. EfficientNet-B0          — 3-phase: feature extraction → partial unfreeze → eval

Metrics computed:
    TOP-1 Accuracy, Precision, Recall, F1-score, G-Mean,
    ROC-AUC (OVR), Confusion Matrix, Parameter count, FLOPs (PyTorch models)

Visualizations:
    t-SNE embedding, Confusion matrices, Loss / Accuracy curves,
    Model comparison bar chart

Usage:
    cd classification
    python split_dataset.py                    # run this first
    python train_classifier.py                  # train all models
    python train_classifier.py --models svm rf  # train only SVM + Random Forest
    python train_classifier.py --use-cv         # use K-Fold CV instead of fixed val split
    python train_classifier.py --no-pca         # disable PCA dimensionality reduction

Output:
    classification/results/
    ├── model_comparison.csv
    ├── model_comparison.json
    ├── confusion_matrices.png
    ├── tsne_embedding.png
    ├── model_comparison.png
    ├── <model>_loss_accuracy.png      (MLP / Transfer Learning)
    ├── svm_report.json
    └── ...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Scikit-learn
# ---------------------------------------------------------------------------
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBClassifier = None

# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import torchvision.models as tv_models

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All metrics we track
ALL_METRIC_KEYS = [
    "top1_accuracy", "precision_macro", "recall_macro", "f1_macro",
    "g_mean", "roc_auc_macro", "roc_auc_weighted",
]


# ===================================================================
# Data loading
# ===================================================================
def load_split_data(data_dir: Path = DATA_DIR):
    """Load the pre-split flat & image data from disk."""
    required = [
        "X_train_flat.npy", "X_val_flat.npy", "X_test_flat.npy",
        "X_train_img.npy", "X_val_img.npy", "X_test_img.npy",
        "y_train.npy", "y_val.npy", "y_test.npy",
    ]
    for fname in required:
        if not (data_dir / fname).exists():
            raise FileNotFoundError(
                f"{fname} not found in {data_dir}. Run split_dataset.py first."
            )

    data = {}
    for split in ["train", "val", "test"]:
        data[f"X_{split}_flat"] = np.load(data_dir / f"X_{split}_flat.npy")
        data[f"X_{split}_img"] = np.load(data_dir / f"X_{split}_img.npy")
        data[f"y_{split}"] = np.load(data_dir / f"y_{split}.npy")

    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            data["metadata"] = json.load(f)

    return data


def load_kfold_data(data_dir: Path = DATA_DIR):
    """Load K-Fold indices if available."""
    kfold_dir = data_dir / "kfold"
    if not kfold_dir.exists():
        return None
    X_train = np.load(data_dir / "X_train_flat.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_train_img = np.load(data_dir / "X_train_img.npy")
    summary_path = kfold_dir / "kfold_summary.json"
    folds = []
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        for fi in range(summary["n_folds"]):
            tr_idx = np.load(kfold_dir / f"fold{fi}_train_idx.npy")
            va_idx = np.load(kfold_dir / f"fold{fi}_val_idx.npy")
            folds.append({"fold": fi, "train_idx": tr_idx, "val_idx": va_idx})
    return {"X_train": X_train, "y_train": y_train, "X_train_img": X_train_img, "folds": folds}


# ===================================================================
# Preprocessing
# ===================================================================
def preprocess_flat(X_train, X_val, X_test, use_pca=True, n_components=0.95):
    """Standardize and optionally apply PCA to flat features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    pca = None
    if use_pca:
        if isinstance(n_components, float):
            pca = PCA(n_components=n_components, random_state=SEED)
        else:
            pca = PCA(n_components=min(n_components, X_train_scaled.shape[0], X_train_scaled.shape[1]),
                      random_state=SEED)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_val_scaled = pca.transform(X_val_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        print(f"  PCA: {X_train_scaled.shape[1]} components (from {X_train.shape[1]})")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler, pca


# ===================================================================
# Metrics  (extended)
# ===================================================================
def g_mean_score(y_true, y_pred):
    """Geometric mean of per-class recall (sensitivity).

    G-Mean = (Π_{c=1..C} recall_c)^(1/C)

    Returns 0 if any class has recall=0.
    """
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    if any(r == 0 for r in recalls):
        return 0.0
    return float(np.prod(recalls) ** (1.0 / len(recalls)))


def evaluate_model(y_true, y_pred, y_score=None, class_names=None):
    """Return a comprehensive dict of metrics.

    Parameters
    ----------
    y_true : (N,) int
    y_pred : (N,) int
    y_score : (N, C) float or None
        Predicted probabilities for each class (used for ROC-AUC).
    class_names : list[str] or None
    """
    metrics = {
        # TOP-1 Accuracy (standard accuracy for single-label classification)
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "g_mean": g_mean_score(y_true, y_pred),
        "precision_per_class": precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # ROC-AUC (One-vs-Rest)
    n_classes = len(np.unique(y_true))
    if y_score is not None and n_classes > 1:
        try:
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
            metrics["roc_auc_macro"] = float(roc_auc_score(
                y_true_bin, y_score, average="macro", multi_class="ovr",
            ))
            metrics["roc_auc_weighted"] = float(roc_auc_score(
                y_true_bin, y_score, average="weighted", multi_class="ovr",
            ))
            # Per-class AUC
            if n_classes == 2:
                metrics["roc_auc_per_class"] = [metrics["roc_auc_macro"]]
            else:
                metrics["roc_auc_per_class"] = [
                    float(roc_auc_score(y_true_bin[:, i], y_score[:, i]))
                    for i in range(n_classes)
                ]
        except Exception:
            metrics["roc_auc_macro"] = None
            metrics["roc_auc_weighted"] = None
            metrics["roc_auc_per_class"] = []
    else:
        metrics["roc_auc_macro"] = None
        metrics["roc_auc_weighted"] = None
        metrics["roc_auc_per_class"] = []

    if class_names:
        metrics["classification_report"] = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0,
        )

    return metrics


def print_metrics(name, metrics):
    """Pretty-print evaluation metrics."""
    print(f"\n  --- {name} ---")
    print(f"  TOP-1 Accuracy:  {metrics['top1_accuracy']:.4f}")
    print(f"  Precision:       {metrics['precision_macro']:.4f} (macro)")
    print(f"  Recall:          {metrics['recall_macro']:.4f} (macro)")
    print(f"  F1-score:        {metrics['f1_macro']:.4f} (macro)")
    print(f"  G-Mean:          {metrics['g_mean']:.4f}")
    if metrics.get("roc_auc_macro") is not None:
        print(f"  ROC-AUC:         {metrics['roc_auc_macro']:.4f} (macro), "
              f"{metrics['roc_auc_weighted']:.4f} (weighted)")
    if "classification_report" in metrics:
        print(f"\n{metrics['classification_report']}")


# ===================================================================
# Parameter counting  &  FLOPs estimation
# ===================================================================
def count_sklearn_params(model, model_name, input_dim=None):
    """Estimate 'effective parameters' for sklearn-style models."""
    est = {}
    try:
        if model_name == "SVM (RBF)":
            n_sv = len(model.support_vectors_)
            est["n_support_vectors"] = n_sv
            est["effective_params"] = n_sv * input_dim if input_dim else n_sv
        elif model_name == "Random Forest":
            total_nodes = sum(
                (t.tree_.node_count if hasattr(t, "tree_") else 0)
                for t in model.estimators_
            )
            est["n_estimators"] = len(model.estimators_)
            est["total_nodes"] = total_nodes
            est["effective_params"] = total_nodes * 2  # threshold + feature per split
        elif model_name == "XGBoost":
            # Rough: number of trees × max_depth leaves
            booster = model.get_booster()
            dump = booster.get_dump()
            est["n_trees"] = len(dump)
            est["effective_params"] = sum(
                len([l for l in tree.split("\n") if "leaf" in l]) for tree in dump
            ) * 2
        elif model_name == "LDA":
            est["n_components"] = model.n_features_in_ * (len(model.classes_) - 1)
            est["effective_params"] = est["n_components"] + len(model.classes_)
    except Exception:
        est["effective_params"] = "N/A"
    return est


def count_torch_params(model):
    """Count trainable and total parameters of a PyTorch model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_params": trainable, "total_params": total}


def estimate_torch_flops(model, input_shape, device=None):
    """Estimate FLOPs for a PyTorch model via forward hooks.

    Covers: Linear, Conv2d, BatchNorm1d, BatchNorm2d, ReLU, Dropout (free).
    Returns total multiply–add operations for one forward pass.
    """
    if device is None:
        device = DEVICE

    flop_counts: dict[str, int] = defaultdict(int)
    handles = []

    def _conv2d_hook(m, inp, out):
        # FLOPs = 2 * batch * Cout * Hout * Wout * (Cin * Kh * Kw // groups)
        x = inp[0]
        batch = x.shape[0]
        cout, cin_per_group, kh, kw = m.weight.shape
        groups = m.groups
        if hasattr(out, "shape"):
            _, _, hout, wout = out.shape
        else:
            hout, wout = 1, 1
        flops = 2 * batch * cout * hout * wout * cin_per_group * kh * kw
        flop_counts[m.__class__.__name__] += flops

    def _linear_hook(m, inp, out):
        x = inp[0]
        batch = x.shape[0] if x.dim() >= 2 else 1
        # FLOPs = 2 * batch * in_features * out_features  (+ out_features for bias)
        flops = 2 * batch * m.in_features * m.out_features
        if m.bias is not None:
            flops += batch * m.out_features
        flop_counts[m.__class__.__name__] += flops

    def _bn_hook(m, inp, out):
        x = inp[0]
        num_elements = x.numel()
        # 2 ops per element: subtract mean (1) + multiply by weight (1)
        # actually: (x - mean) / sqrt(var+eps) * weight + bias  →  ~5 ops
        flop_counts[m.__class__.__name__] += num_elements * 5

    def _relu_hook(m, inp, out):
        flop_counts[m.__class__.__name__] += inp[0].numel()

    def _adaptive_pool_hook(m, inp, out):
        flop_counts[m.__class__.__name__] += inp[0].numel()

    hook_map = {
        nn.Conv2d: _conv2d_hook,
        nn.Linear: _linear_hook,
        nn.BatchNorm1d: _bn_hook,
        nn.BatchNorm2d: _bn_hook,
        nn.ReLU: _relu_hook,
        nn.AdaptiveAvgPool2d: _adaptive_pool_hook,
    }

    for module in model.modules():
        for cls, hook_fn in hook_map.items():
            if isinstance(module, cls):
                handles.append(module.register_forward_hook(hook_fn))
                break

    # Run one forward pass with dummy input
    model.eval()
    dummy = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        try:
            model(dummy)
        except Exception:
            pass

    for h in handles:
        h.remove()

    total_flops = sum(flop_counts.values())
    flop_counts["total"] = total_flops
    return dict(flop_counts)


# ===================================================================
# Visualization
# ===================================================================
def plot_tsne(X, y, class_names, title="t-SNE Embedding",
              save_path=None, ax=None, pred_labels=None):
    """Plot 2D t-SNE of a feature space.

    Parameters
    ----------
    X : (N, D) array  — feature vectors.
    y : (N,) array    — ground-truth integer labels.
    class_names : list[str]
    title : str
    save_path : Path or None  — if given, save the figure.
    ax : matplotlib Axes or None  — if given, draw into this subplot.
    pred_labels : (N,) array or None  — if given, overlay a second row
        of scatter points showing predicted labels (useful for sklearn
        models that operate directly on input space).
    """
    if not HAS_PLOTTING:
        return None

    max_samples = 1000
    if X.shape[0] > max_samples:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X_sub, y_sub = X[idx], y[idx]
        pred_sub = pred_labels[idx] if pred_labels is not None else None
    else:
        X_sub, y_sub = X, y
        pred_sub = pred_labels

    tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, X_sub.shape[0] - 1))
    X_2d = tsne.fit_transform(X_sub)

    colors = sns.color_palette("husl", len(class_names))

    # --- Determine layout ---
    nrows = 2 if pred_sub is not None else 1
    own_fig = ax is None
    if own_fig:
        fig, axes = plt.subplots(nrows, 1, figsize=(8, 5 * nrows))
        if nrows == 1:
            axes = [axes]
        fig.suptitle(title, fontsize=13, fontweight="bold")
    else:
        axes = [ax] if nrows == 1 else [ax, None]  # placeholder — see below

    # --- Row 1: true labels ---
    ax0 = axes[0]
    for i, name in enumerate(class_names):
        mask = y_sub == i
        ax0.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]], label=name,
                    alpha=0.7, edgecolors="k", linewidth=0.3, s=30)
    ax0.set_title("True Labels" if pred_sub is not None else title, fontsize=10)
    ax0.set_xlabel("t-SNE 1")
    ax0.set_ylabel("t-SNE 2")
    ax0.legend(loc="best", fontsize=7)

    # --- Row 2: predicted labels (if given) ---
    if pred_sub is not None:
        if own_fig:
            ax1 = axes[1]
        else:
            # We need a second subplot — this case is handled by
            # plot_tsne_per_model which passes a pre-created ax pair.
            return X_2d, y_sub, pred_sub  # deferred to caller
        for i, name in enumerate(class_names):
            mask = pred_sub == i
            ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]],
                        alpha=0.7, edgecolors="k", linewidth=0.3, s=30)
        ax1.set_title("Predicted Labels", fontsize=10)
        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")

    if own_fig:
        fig.tight_layout()
        if save_path is None:
            save_path = RESULTS_DIR / "tsne_embedding.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  t-SNE saved to {save_path}")

    return None


def _extract_model_features(model, model_name, X, X_img=None, device=None):
    """Extract a feature representation from a trained model for t-SNE.

    Returns
    -------
    features : (N, D) ndarray  — feature vectors for t-SNE.
    predictions : (N,) ndarray or None
    is_learned : bool  — True if features come from a learned embedding.
    """
    if device is None:
        device = DEVICE

    # -- LDA: has a proper learned transform --
    if model_name == "LDA":
        try:
            feats = model.transform(X)
            preds = model.predict(X)
            return feats, preds, True
        except Exception:
            pass

    # -- PyTorch models: extract penultimate / backbone features --
    if model_name in ("MLP (Small)", "Transfer Learning (ResNet18)",
                       "EfficientNet-B0 (3-Phase)"):
        try:
            model.eval()
            if model_name == "MLP (Small)":
                # model.net is Sequential; penultimate is net[-2] (before final Linear)
                X_t = torch.tensor(X, dtype=torch.float32).to(device)
                penultimate = model.net[:-1]  # all layers except final Linear
                with torch.no_grad():
                    feats = penultimate(X_t).cpu().numpy()
                    logits = model.net(X_t)
                    preds = logits.argmax(dim=1).cpu().numpy()
                return feats, preds, True
            else:
                # CNN models (ResNet18 / EfficientNet-B0)
                X_t = torch.tensor(X_img if X_img is not None else X, dtype=torch.float32)
                # Resize if needed
                if X_t.shape[-2:] != (224, 224):
                    X_t = nn.functional.interpolate(
                        X_t, size=(224, 224), mode="bilinear", align_corners=False,
                    )
                X_t = X_t.to(device)
                with torch.no_grad():
                    logits = model(X_t)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    # For ResNet18: model.backbone(x) gives features
                    # For EfficientNet: model.backbone.features + avgpool + flatten
                    if hasattr(model, "backbone"):
                        if hasattr(model.backbone, "features"):
                            # EfficientNet
                            f = model.backbone.features(X_t)
                            f = model.backbone.avgpool(f)
                            feats = torch.flatten(f, 1).cpu().numpy()
                        else:
                            # ResNet18
                            feats = model.backbone(X_t).cpu().numpy()
                    else:
                        feats = X_t.cpu().numpy().reshape(X_t.shape[0], -1)
                return feats, preds, True
        except Exception as e:
            pass  # fall through to input-features fallback

    # -- Sklearn models without learned embedding (SVM, RF, XGBoost) --
    # Return input features + predictions so we can color by predicted labels.
    try:
        preds = model.predict(X)
        return X, preds, False
    except Exception:
        return X, None, False


def plot_tsne_per_model(all_results, X_test_pp, X_test_img, y_test, class_names,
                        results_dir=None):
    """Grid of t-SNE plots — one row per trained model.

    - Models with learned embeddings (LDA, MLP, CNNs): t-SNE of the
      learned feature space, colored by true labels.
    - Models without (SVM, RF, XGBoost): t-SNE of input features, with
      two rows (true labels / predicted labels) to show model behaviour.
    """
    if not HAS_PLOTTING:
        return

    valid = [r for r in all_results if r is not None]
    if not valid:
        return

    # Build feature representations for each model
    entries = []
    for r in valid:
        mname = r["model"]
        print(f"  Extracting features for {mname} ...")

        # Get the fitted model object
        model = r.get("_fitted_model", None)

        # For PyTorch models stored in _fitted_model, use them directly
        # (they've already been moved to CPU in the training function)
        feats, preds, is_learned = _extract_model_features(
            model, mname, X_test_pp, X_test_img,
        )
        entries.append({
            "name": mname,
            "features": feats,
            "predictions": preds,
            "is_learned": is_learned,
        })

    # ---- Plot grid ----
    n = len(entries)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows),
                              squeeze=False)
    colors = sns.color_palette("husl", len(class_names))

    for idx, entry in enumerate(entries):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]

        X = entry["features"]
        y = y_test[:len(X)] if len(y_test) >= len(X) else np.tile(y_test, 2)[:len(X)]
        preds = entry["predictions"]
        is_learned = entry["is_learned"]

        # Subsample
        max_samples = 500
        if X.shape[0] > max_samples:
            rng = np.random.RandomState(SEED)
            sub_idx = rng.choice(X.shape[0], max_samples, replace=False)
            X_sub = X[sub_idx]
            y_sub = y[sub_idx]
            pred_sub = preds[sub_idx] if preds is not None else None
        else:
            X_sub, y_sub = X, y
            pred_sub = preds

        tsne = TSNE(n_components=2, random_state=SEED,
                    perplexity=min(30, X_sub.shape[0] - 1))
        X_2d = tsne.fit_transform(X_sub)

        for i, name in enumerate(class_names):
            mask = y_sub == i
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]], label=name,
                       alpha=0.7, edgecolors="k", linewidth=0.3, s=35)

        subtitle = f"{entry['name']}"
        if is_learned:
            subtitle += "\n(learned embedding)"
        else:
            subtitle += "\n(input space)"
        ax.set_title(subtitle, fontsize=10, fontweight="bold")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        if idx == 0:
            ax.legend(loc="best", fontsize=6)

    # Hide unused
    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].set_visible(False)

    fig.suptitle("t-SNE — Per-Model Feature Representations (colored by true labels)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    save_path = (results_dir or RESULTS_DIR) / "tsne_per_model.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nt-SNE per model saved to {save_path}")


def plot_confusion_matrices(all_results, class_names):
    """Grid of row-normalized confusion matrices for all trained models."""
    if not HAS_PLOTTING:
        print("\n[SKIP] matplotlib/seaborn not available for plotting.")
        return

    valid = [(r["model"], r["test"]["confusion_matrix"]) for r in all_results if r is not None]
    if not valid:
        return

    n = len(valid)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (model_name, cm) in enumerate(valid):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        cm_arr = np.array(cm, dtype=float)
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_arr / row_sums
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0, vmax=1, ax=ax, cbar=(idx == n - 1))
        ax.set_title(model_name, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].set_visible(False)

    fig.suptitle("Confusion Matrices (Row-Normalized)", fontsize=14, fontweight="bold")
    fig.tight_layout()

    plot_path = RESULTS_DIR / "confusion_matrices.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nConfusion matrices saved to {plot_path}")


def plot_loss_accuracy_curves(history, model_name, save_path=None):
    """Plot training & validation loss and accuracy curves from logged history.

    Parameters
    ----------
    history : dict
        Keys: ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc`` — each a list.
    """
    if not HAS_PLOTTING or not history:
        return

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    if len(epochs) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=1.5)
    if history.get("val_loss"):
        ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} — Loss Curve")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Accuracy", linewidth=1.5)
    if history.get("val_acc"):
        ax2.plot(epochs, history["val_acc"], "r-", label="Val Accuracy", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model_name} — Accuracy Curve")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(model_name, fontweight="bold")
    fig.tight_layout()

    if save_path is None:
        slug = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_path = RESULTS_DIR / f"{slug}_loss_accuracy.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss/accuracy curves saved to {save_path}")


def plot_model_comparison(all_results):
    """Grouped bar chart of all metrics across models."""
    if not HAS_PLOTTING:
        return

    valid = [r for r in all_results if r is not None]
    if not valid:
        return

    models = [r["model"] for r in valid]
    metric_keys = ["top1_accuracy", "precision_macro", "recall_macro", "f1_macro", "g_mean"]
    display_names = ["TOP-1 Acc", "Precision", "Recall", "F1", "G-Mean"]
    # Filter to metrics that exist
    available = []
    available_names = []
    for mk, dn in zip(metric_keys, display_names):
        if all(mk in r["test"] for r in valid):
            available.append(mk)
            available_names.append(dn)

    x = np.arange(len(models))
    width = 0.16
    n_metrics = len(available)

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(models)), 5.5))
    palette = sns.color_palette("Set2", n_metrics)

    for i, (mkey, mlabel) in enumerate(zip(available, available_names)):
        values = [r["test"][mkey] for r in valid]
        offset = (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=mlabel, color=palette[i])
        for bar, val in zip(bars, values):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set Metrics")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    plot_path = RESULTS_DIR / "model_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison chart saved to {plot_path}")


# ===================================================================
# 1. SVM (RBF kernel)
# ===================================================================
def train_svm(X_train, y_train, X_val, y_val, X_test, y_test, class_names,
              use_cv=False, cv_folds=5):
    print("\n" + "=" * 60)
    print("1. SVM (RBF Kernel)")
    print("=" * 60)

    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1, 1],
        "kernel": ["rbf"],
    }

    if use_cv:
        inner_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
        grid = GridSearchCV(
            SVC(probability=True, random_state=SEED, class_weight="balanced"),
            param_grid, cv=inner_cv, scoring="f1_macro",
            n_jobs=-1, verbose=0, refit=True,
        )
        X_all = np.concatenate([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        grid.fit(X_all, y_all)
        best_model = grid.best_estimator_
        y_pred_test = best_model.predict(X_test)
        y_prob_test = best_model.predict_proba(X_test)
        test_metrics = evaluate_model(y_test, y_pred_test, y_prob_test, class_names)
        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV score: {grid.best_score_:.4f}")
    else:
        grid = GridSearchCV(
            SVC(probability=True, random_state=SEED, class_weight="balanced"),
            param_grid, cv=3, scoring="f1_macro",
            n_jobs=-1, verbose=0, refit=True,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred_val = best_model.predict(X_val)
        y_prob_val = best_model.predict_proba(X_val)
        val_metrics = evaluate_model(y_val, y_pred_val, y_prob_val, class_names)
        print(f"  Best params: {grid.best_params_}")
        print_metrics("Validation", val_metrics)

        y_pred_test = best_model.predict(X_test)
        y_prob_test = best_model.predict_proba(X_test)
        test_metrics = evaluate_model(y_test, y_pred_test, y_prob_test, class_names)

    print_metrics("Test", test_metrics)

    # Parameter count
    params_info = count_sklearn_params(best_model, "SVM (RBF)", X_train.shape[1])
    print(f"  Support vectors: {params_info.get('n_support_vectors', 'N/A')}")
    print(f"  Effective params: {params_info.get('effective_params', 'N/A')}")

    return {
        "model": "SVM (RBF)",
        "test": test_metrics,
        "best_params": grid.best_params_,
        "param_info": params_info,
        "flops": "N/A (non-parametric inference)",
        "_fitted_model": best_model,
    }


# ===================================================================
# 2. Random Forest
# ===================================================================
def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, class_names,
                        use_cv=False, cv_folds=5):
    print("\n" + "=" * 60)
    print("2. Random Forest")
    print("=" * 60)

    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    if use_cv:
        inner_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
        grid = GridSearchCV(
            RandomForestClassifier(random_state=SEED, class_weight="balanced", n_jobs=-1),
            param_grid, cv=inner_cv, scoring="f1_macro",
            n_jobs=-1, verbose=0, refit=True,
        )
        X_all = np.concatenate([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        grid.fit(X_all, y_all)
        best_model = grid.best_estimator_
        y_pred_test = best_model.predict(X_test)
        y_prob_test = best_model.predict_proba(X_test)
        test_metrics = evaluate_model(y_test, y_pred_test, y_prob_test, class_names)
        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV score: {grid.best_score_:.4f}")
    else:
        grid = GridSearchCV(
            RandomForestClassifier(random_state=SEED, class_weight="balanced", n_jobs=-1),
            param_grid, cv=3, scoring="f1_macro",
            n_jobs=-1, verbose=0, refit=True,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred_val = best_model.predict(X_val)
        y_prob_val = best_model.predict_proba(X_val)
        val_metrics = evaluate_model(y_val, y_pred_val, y_prob_val, class_names)
        print(f"  Best params: {grid.best_params_}")
        print_metrics("Validation", val_metrics)

        y_pred_test = best_model.predict(X_test)
        y_prob_test = best_model.predict_proba(X_test)
        test_metrics = evaluate_model(y_test, y_pred_test, y_prob_test, class_names)

    print_metrics("Test", test_metrics)

    # Feature importance
    importances = best_model.feature_importances_
    top_n = min(10, len(importances))
    top_idx = np.argsort(importances)[::-1][:top_n]
    print(f"\n  Top {top_n} feature importances:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"    {rank}. Feature {idx}: {importances[idx]:.4f}")

    params_info = count_sklearn_params(best_model, "Random Forest", X_train.shape[1])
    print(f"  Trees: {params_info.get('n_estimators', 'N/A')}, "
          f"Total nodes: {params_info.get('total_nodes', 'N/A')}")

    return {
        "model": "Random Forest",
        "test": test_metrics,
        "best_params": grid.best_params_,
        "param_info": params_info,
        "flops": "N/A",
        "_fitted_model": best_model,
    }


# ===================================================================
# 3. XGBoost
# ===================================================================
def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, class_names,
                  use_cv=False, cv_folds=5):
    print("\n" + "=" * 60)
    print("3. XGBoost")
    print("=" * 60)

    if not HAS_XGBOOST:
        print("  [SKIP] xgboost not installed.  pip install xgboost")
        return None

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    if use_cv:
        inner_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
        grid = GridSearchCV(
            XGBClassifier(random_state=SEED, eval_metric="mlogloss", verbosity=0),
            param_grid, cv=inner_cv, scoring="f1_macro",
            n_jobs=-1, verbose=0, refit=True,
        )
        X_all = np.concatenate([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        grid.fit(X_all, y_all)
        best_model = grid.best_estimator_
        y_pred_test = best_model.predict(X_test)
        y_prob_test = best_model.predict_proba(X_test)
        test_metrics = evaluate_model(y_test, y_pred_test, y_prob_test, class_names)
        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV score: {grid.best_score_:.4f}")
    else:
        grid = GridSearchCV(
            XGBClassifier(random_state=SEED, eval_metric="mlogloss", verbosity=0),
            param_grid, cv=3, scoring="f1_macro",
            n_jobs=-1, verbose=0, refit=True,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred_val = best_model.predict(X_val)
        y_prob_val = best_model.predict_proba(X_val)
        val_metrics = evaluate_model(y_val, y_pred_val, y_prob_val, class_names)
        print(f"  Best params: {grid.best_params_}")
        print_metrics("Validation", val_metrics)

        y_pred_test = best_model.predict(X_test)
        y_prob_test = best_model.predict_proba(X_test)
        test_metrics = evaluate_model(y_test, y_pred_test, y_prob_test, class_names)

    print_metrics("Test", test_metrics)

    params_info = count_sklearn_params(best_model, "XGBoost", X_train.shape[1])
    print(f"  Trees: {params_info.get('n_trees', 'N/A')}, "
          f"Effective params: {params_info.get('effective_params', 'N/A')}")

    return {
        "model": "XGBoost",
        "test": test_metrics,
        "best_params": grid.best_params_,
        "param_info": params_info,
        "flops": "N/A",
        "_fitted_model": best_model,
    }


# ===================================================================
# 4. LDA
# ===================================================================
def train_lda(X_train, y_train, X_val, y_val, X_test, y_test, class_names,
              use_cv=False, cv_folds=5):
    print("\n" + "=" * 60)
    print("4. Linear Discriminant Analysis (LDA)")
    print("=" * 60)

    n_components_max = len(class_names) - 1

    if use_cv:
        X_all = np.concatenate([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])

        best_score = -1
        best_shrinkage = None
        for shrinkage in [None, "auto", 0.1, 0.5, 1.0]:
            try:
                solver = "eigen" if shrinkage else "svd"
                lda = LinearDiscriminantAnalysis(
                    n_components=n_components_max, shrinkage=shrinkage, solver=solver,
                )
                scores = cross_val_score(lda, X_all, y_all, cv=cv_folds, scoring="f1_macro")
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_shrinkage = shrinkage
            except Exception:
                continue

        solver = "eigen" if best_shrinkage else "svd"
        lda = LinearDiscriminantAnalysis(
            n_components=n_components_max, shrinkage=best_shrinkage, solver=solver,
        )
        lda.fit(X_all, y_all)
        y_pred_test = lda.predict(X_test)
        y_prob_test = lda.predict_proba(X_test)
        test_metrics = evaluate_model(y_test, y_pred_test, y_prob_test, class_names)
        print(f"  Shrinkage: {best_shrinkage}")
    else:
        best_val_f1 = -1
        best_lda = None
        best_shrinkage = None
        for shrinkage in [None, "auto", 0.1, 0.5, 1.0]:
            try:
                solver = "eigen" if shrinkage else "svd"
                lda = LinearDiscriminantAnalysis(
                    n_components=n_components_max, shrinkage=shrinkage, solver=solver,
                )
                lda.fit(X_train, y_train)
                y_pred_val = lda.predict(X_val)
                val_f1 = f1_score(y_val, y_pred_val, average="macro", zero_division=0)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_lda = lda
                    best_shrinkage = shrinkage
            except Exception:
                continue

        if best_lda is None:
            print("  [ERROR] LDA failed for all shrinkage values.")
            return None

        y_pred_val = best_lda.predict(X_val)
        y_prob_val = best_lda.predict_proba(X_val)
        val_metrics = evaluate_model(y_val, y_pred_val, y_prob_val, class_names)
        print(f"  Shrinkage: {best_shrinkage}")
        print_metrics("Validation", val_metrics)

        y_pred_test = best_lda.predict(X_test)
        y_prob_test = best_lda.predict_proba(X_test)
        test_metrics = evaluate_model(y_test, y_pred_test, y_prob_test, class_names)

    print_metrics("Test", test_metrics)

    params_info = count_sklearn_params(best_lda if not use_cv else lda, "LDA", X_train.shape[1])
    print(f"  Effective params: {params_info.get('effective_params', 'N/A')}")

    fitted = best_lda if not use_cv else lda
    return {
        "model": "LDA",
        "test": test_metrics,
        "best_params": {"shrinkage": best_shrinkage},
        "param_info": params_info,
        "flops": "N/A",
        "_fitted_model": fitted,
    }


# ===================================================================
# 5. MLP (small, PyTorch)  — with loss / acc logging
# ===================================================================
class SmallMLP(nn.Module):
    """A small MLP with strong regularization for small datasets."""

    def __init__(self, input_dim, num_classes, hidden_dims=(128, 64), dropout=0.5):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def _eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
        all_probs.append(probs.cpu())
        all_labels.append(yb.cpu())
    return (total_loss / total, correct / total,
            torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy())


def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, class_names,
              use_cv=False, cv_folds=5):
    print("\n" + "=" * 60)
    print("5. MLP (Small, Regularized)")
    print("=" * 60)

    if not HAS_TORCH:
        print("  [SKIP] PyTorch not installed.")
        return None

    input_dim = X_train.shape[1]
    num_classes = len(class_names)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    configs = [
        {"hidden_dims": (64, 32), "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-3},
        {"hidden_dims": (128, 64), "dropout": 0.5, "lr": 1e-3, "weight_decay": 1e-3},
        {"hidden_dims": (64,), "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-2},
        {"hidden_dims": (128, 64, 32), "dropout": 0.5, "lr": 1e-4, "weight_decay": 1e-2},
    ]

    best_val_f1 = -1
    best_model_state = None
    best_config = None
    best_history = None

    for cfg in configs:
        torch.manual_seed(SEED)
        model = SmallMLP(input_dim, num_classes, cfg["hidden_dims"], cfg["dropout"]).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        for epoch in range(100):
            tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, criterion)
            va_loss, va_acc, va_prob, va_lab = _eval_epoch(model, val_loader, criterion)
            scheduler.step()
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)

        va_f1 = f1_score(va_lab, va_prob.argmax(axis=1), average="macro", zero_division=0)
        print(f"  Config {cfg}: val_f1={va_f1:.4f}, val_acc={history['val_acc'][-1]:.4f}")
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_config = cfg
            best_history = history

    print(f"\n  Best config: {best_config}, val_f1={best_val_f1:.4f}")

    # Retrain best config on train+val
    X_combined = torch.cat([X_train_t, X_val_t], dim=0)
    y_combined = torch.cat([y_train_t, y_val_t], dim=0)
    combined_ds = TensorDataset(X_combined, y_combined)
    combined_loader = DataLoader(combined_ds, batch_size=32, shuffle=True)

    torch.manual_seed(SEED)
    final_model = SmallMLP(input_dim, num_classes, best_config["hidden_dims"], best_config["dropout"]).to(DEVICE)
    optimizer = optim.AdamW(final_model.parameters(), lr=best_config["lr"],
                             weight_decay=best_config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    criterion = nn.CrossEntropyLoss()

    final_history = {"train_loss": [], "train_acc": []}
    for epoch in range(150):
        tr_loss, tr_acc = _train_epoch(final_model, combined_loader, optimizer, criterion)
        scheduler.step()
        final_history["train_loss"].append(tr_loss)
        final_history["train_acc"].append(tr_acc)

    # Test
    final_model.eval()
    with torch.no_grad():
        logits = final_model(X_test_t.to(DEVICE))
        y_prob_test = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred_test = logits.argmax(dim=1).cpu().numpy()

    test_metrics = evaluate_model(y_test, y_pred_test, y_prob_test, class_names)
    print_metrics("Test", test_metrics)

    # Parameter count & FLOPs
    params_info = count_torch_params(final_model)
    print(f"  Trainable params: {params_info['trainable_params']:,}, "
          f"Total params: {params_info['total_params']:,}")
    flops_info = estimate_torch_flops(final_model, (1, input_dim))
    print(f"  FLOPs (forward): {flops_info.get('total', 'N/A'):,}")

    # Save model
    model_path = RESULTS_DIR / "mlp_model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": final_model.state_dict(), "config": best_config,
                "input_dim": input_dim, "num_classes": num_classes}, model_path)
    print(f"  Model saved to {model_path}")

    # Plot loss / accuracy curves
    plot_loss_accuracy_curves(best_history, "MLP (Small)",
                              save_path=RESULTS_DIR / "mlp_loss_accuracy.png")

    return {
        "model": "MLP (Small)",
        "test": test_metrics,
        "best_params": best_config,
        "param_info": params_info,
        "flops": flops_info,
        "history": best_history,
        "_fitted_model": final_model.cpu(),
    }


# ===================================================================
# 6. Transfer Learning (CNN — ResNet18)
# ===================================================================
class SpectrogramClassifier(nn.Module):
    """Transfer learning wrapper: pretrained ResNet backbone → classifier head."""

    def __init__(self, num_classes, backbone="resnet18", freeze_backbone=True):
        super().__init__()
        if backbone == "resnet18":
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = tv_models.resnet18(weights=weights)
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False,
            )
            with torch.no_grad():
                self.backbone.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
            for param in self.backbone.conv1.parameters():
                param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ===================================================================
# 7. EfficientNet-B0  —  3-phase fine-tuning
# ===================================================================
class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 backbone → custom classifier head for 1-channel input.

    The pretrained features are grouped into stages.  During Phase 2 we
    unfreeze only the last few stages to avoid catastrophic forgetting on
    a small dataset.
    """

    # Stage boundaries in torchvision's efficientnet_b0 features Sequential.
    # Index 0 = stem conv, 1..7 = MBConv stages, 8 = final Conv+BN+SiLU.
    _FEATURES_TOTAL = 9   # features[0] … features[8]

    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = tv_models.efficientnet_b0(weights=weights)

        # --- Replace first conv to accept 1 channel instead of 3 ---
        old_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        self.backbone.features[0][0] = new_conv

        # --- Replace classifier head ---
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

        # --- Freeze / partial-freeze ---
        if freeze_backbone:
            self._set_backbone_requires_grad(False)
            # Unfreeze the new conv1 so it can adapt
            for p in self.backbone.features[0].parameters():
                p.requires_grad = True

    def _set_backbone_requires_grad(self, requires_grad: bool):
        for p in self.backbone.parameters():
            p.requires_grad = requires_grad

    def unfreeze_last_stages(self, num_stages: int = 3):
        """Unfreeze the last *num_stages* MBConv stages (features[6..8] by default).

        EfficientNet-B0 features layout (torchvision):
            features[0]  — stem Conv2d + BN + SiLU
            features[1]  — MBConv stage 1  (1 block)
            features[2]  — MBConv stage 2  (2 blocks)
            features[3]  — MBConv stage 3  (2 blocks)
            features[4]  — MBConv stage 4  (3 blocks)
            features[5]  — MBConv stage 5  (3 blocks)
            features[6]  — MBConv stage 6  (4 blocks)
            features[7]  — MBConv stage 7  (1 block)
            features[8]  — final Conv2d + BN + SiLU
        """
        start_idx = max(1, self._FEATURES_TOTAL - num_stages)
        for idx in range(start_idx, self._FEATURES_TOTAL):
            for p in self.backbone.features[idx].parameters():
                p.requires_grad = True

    def forward(self, x):
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        return self.classifier(features)


def _resize_batch(batch):
    return nn.functional.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)


def _train_epoch_cnn(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb = _resize_batch(xb.to(DEVICE))
        yb = yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def _eval_epoch_cnn(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for xb, yb in loader:
        xb = _resize_batch(xb.to(DEVICE))
        yb = yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
        all_probs.append(probs.cpu())
        all_labels.append(yb.cpu())
    return (total_loss / total, correct / total,
            torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy())


def train_transfer_learning(X_train_img, y_train, X_val_img, y_val, X_test_img, y_test, class_names):
    print("\n" + "=" * 60)
    print("6. Transfer Learning (ResNet18)")
    print("=" * 60)

    if not HAS_TORCH or not HAS_TORCHVISION:
        print("  [SKIP] PyTorch / torchvision not installed.")
        return None

    num_classes = len(class_names)

    X_train_t = torch.tensor(X_train_img, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val_img, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test_img, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    configs = [
        {"freeze": True, "lr": 1e-3, "label": "frozen_backbone"},
        {"freeze": False, "lr": 1e-4, "label": "fine_tuned"},
    ]

    best_val_f1 = -1
    best_model_state = None
    best_config_label = None
    best_history = None

    for cfg in configs:
        torch.manual_seed(SEED)
        model = SpectrogramClassifier(num_classes, freeze_backbone=cfg["freeze"]).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        for epoch in range(50):
            tr_loss, tr_acc = _train_epoch_cnn(model, train_loader, optimizer, criterion)
            va_loss, va_acc, va_prob, va_lab = _eval_epoch_cnn(model, val_loader, criterion)
            scheduler.step()
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)

        va_f1 = f1_score(va_lab, va_prob.argmax(axis=1), average="macro", zero_division=0)
        print(f"  {cfg['label']}: val_f1={va_f1:.4f}, val_acc={history['val_acc'][-1]:.4f}")
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_config_label = cfg["label"]
            best_history = history

    if best_model_state is None:
        print("  [ERROR] Transfer learning failed.")
        return None

    print(f"\n  Best config: {best_config_label}, val_f1={best_val_f1:.4f}")
    best_freeze = "frozen" in best_config_label

    X_combined = torch.cat([X_train_t, X_val_t], dim=0)
    y_combined = torch.cat([y_train_t, y_val_t], dim=0)
    combined_ds = TensorDataset(X_combined, y_combined)
    combined_loader = DataLoader(combined_ds, batch_size=16, shuffle=True)

    torch.manual_seed(SEED)
    final_model = SpectrogramClassifier(num_classes, freeze_backbone=best_freeze).to(DEVICE)
    lr = 1e-3 if best_freeze else 1e-4
    optimizer = optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    criterion = nn.CrossEntropyLoss()

    final_history = {"train_loss": [], "train_acc": []}
    for epoch in range(80):
        tr_loss, tr_acc = _train_epoch_cnn(final_model, combined_loader, optimizer, criterion)
        scheduler.step()
        final_history["train_loss"].append(tr_loss)
        final_history["train_acc"].append(tr_acc)

    # Test
    final_model.eval()
    with torch.no_grad():
        logits = final_model(_resize_batch(X_test_t.to(DEVICE)))
        y_prob_test = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred_test = logits.argmax(dim=1).cpu().numpy()

    test_metrics = evaluate_model(y_test, y_pred_test, y_prob_test, class_names)
    print_metrics("Test", test_metrics)

    # Parameter count & FLOPs
    params_info = count_torch_params(final_model)
    print(f"  Trainable params: {params_info['trainable_params']:,}, "
          f"Total params: {params_info['total_params']:,}")
    flops_info = estimate_torch_flops(final_model, (1, 1, 224, 224))
    print(f"  FLOPs (forward): {flops_info.get('total', 'N/A'):,}")

    # Save model
    model_path = RESULTS_DIR / "transfer_learning_model.pt"
    torch.save({"state_dict": final_model.state_dict(), "config": best_config_label,
                "num_classes": num_classes}, model_path)
    print(f"  Model saved to {model_path}")

    # Plot loss / accuracy curves
    plot_loss_accuracy_curves(best_history, "Transfer Learning (ResNet18)",
                              save_path=RESULTS_DIR / "transfer_learning_loss_accuracy.png")

    return {
        "model": "Transfer Learning (ResNet18)",
        "test": test_metrics,
        "best_params": {"freeze_backbone": best_freeze},
        "param_info": params_info,
        "flops": flops_info,
        "history": best_history,
        "_fitted_model": final_model.cpu(),
    }


# ===================================================================
# 7. EfficientNet-B0  —  3-phase fine-tuning
# ===================================================================
def _build_diff_param_groups(model, head_lr, backbone_lr):
    """Create per-layer parameter groups for differential learning rates.

    Parameters in ``model.classifier`` get ``head_lr``; all other trainable
    parameters get ``backbone_lr``.
    """
    head_ids = {id(p) for p in model.classifier.parameters()}
    head_params = []
    backbone_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head_params.append(p)
        else:
            backbone_params.append(p)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    return groups


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # improved
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False  # did not improve


def train_efficientnet(X_train_img, y_train, X_val_img, y_val, X_test_img, y_test, class_names):
    """3-phase fine-tuning of EfficientNet-B0.

    Phase 1 — Feature Extraction (epochs  1–10)
        Freeze ALL pretrained layers, train only the new classifier head.
        Higher LR: 1e-3.

    Phase 2 — Partial Unfreezing (epochs 11–30)
        Unfreeze last 3 MBConv stages.
        Differential LR: head=1e-4, backbone=1e-5.

    Phase 3 — Evaluation
        Early stopping (patience=8) monitors validation loss.
    """
    print("\n" + "=" * 60)
    print("7. EfficientNet-B0 (3-Phase Fine-Tuning)")
    print("=" * 60)

    if not HAS_TORCH or not HAS_TORCHVISION:
        print("  [SKIP] PyTorch / torchvision not installed.")
        return None

    # Check EfficientNet is available in this torchvision version
    try:
        _ = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
    except AttributeError:
        print("  [SKIP] EfficientNet not available in this torchvision version "
              "(requires torchvision >= 0.11).")
        return None

    num_classes = len(class_names)

    # --- Tensors ---
    X_train_t = torch.tensor(X_train_img, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val_img, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test_img, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
    }

    # ---- Build model ----
    torch.manual_seed(SEED)
    model = EfficientNetB0Classifier(num_classes, freeze_backbone=True).to(DEVICE)

    # ==================================================================
    # Phase 1 — Feature Extraction (epochs 1–10)
    # ==================================================================
    print("\n  [Phase 1] Feature Extraction (epochs  1–10)")
    print("            Freeze backbone → train classifier head only,  lr=1e-3")

    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    early_stop = EarlyStopping(patience=8)

    for epoch in range(1, 11):
        tr_loss, tr_acc = _train_epoch_cnn(model, train_loader, optimizer, criterion)
        va_loss, va_acc, va_prob, va_lab = _eval_epoch_cnn(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        improved = early_stop.step(va_loss)
        marker = " *" if improved else ""
        print(f"    Epoch {epoch:2d}:  tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}  "
              f"va_acc={va_acc:.4f}{marker}")
        if early_stop.should_stop:
            print(f"    Early stopping at epoch {epoch}")
            break

    # ==================================================================
    # Phase 2 — Partial Unfreezing (epochs 11–30)
    # ==================================================================
    print("\n  [Phase 2] Partial Unfreezing (epochs 11–30)")
    print("            Unfreeze last 3 stages → differential LR:  head=1e-4,  backbone=1e-5")

    model.unfreeze_last_stages(num_stages=3)
    param_groups = _build_diff_param_groups(model, head_lr=1e-4, backbone_lr=1e-5)
    optimizer = optim.AdamW(param_groups, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    early_stop = EarlyStopping(patience=8)

    for epoch in range(11, 31):
        tr_loss, tr_acc = _train_epoch_cnn(model, train_loader, optimizer, criterion)
        va_loss, va_acc, va_prob, va_lab = _eval_epoch_cnn(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        improved = early_stop.step(va_loss)
        marker = " *" if improved else ""
        print(f"    Epoch {epoch:2d}:  tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}  "
              f"va_acc={va_acc:.4f}{marker}")
        if early_stop.should_stop:
            print(f"    Early stopping at epoch {epoch}")
            break

    # ---- Retrain best config on train+val combined for final eval ----
    print("\n  [Final] Retraining best strategy on train+val combined...")

    X_combined = torch.cat([X_train_t, X_val_t], dim=0)
    y_combined = torch.cat([y_train_t, y_val_t], dim=0)
    combined_ds = TensorDataset(X_combined, y_combined)
    combined_loader = DataLoader(combined_ds, batch_size=16, shuffle=True)

    # Rebuild and repeat the 2-phase protocol on combined data
    torch.manual_seed(SEED)
    final_model = EfficientNetB0Classifier(num_classes, freeze_backbone=True).to(DEVICE)

    # Phase 1 on combined: 15 epochs
    opt = optim.AdamW(final_model.classifier.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
    for epoch in range(15):
        _train_epoch_cnn(final_model, combined_loader, opt, criterion)
        sched.step()

    # Phase 2 on combined: 25 epochs with differential LR
    final_model.unfreeze_last_stages(num_stages=3)
    pg = _build_diff_param_groups(final_model, head_lr=1e-4, backbone_lr=1e-5)
    opt = optim.AdamW(pg, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
    for epoch in range(25):
        _train_epoch_cnn(final_model, combined_loader, opt, criterion)
        sched.step()

    # ---- Test ----
    final_model.eval()
    with torch.no_grad():
        logits = final_model(_resize_batch(X_test_t.to(DEVICE)))
        y_prob_test = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred_test = logits.argmax(dim=1).cpu().numpy()

    test_metrics = evaluate_model(y_test, y_pred_test, y_prob_test, class_names)
    print_metrics("Test", test_metrics)

    # Parameter count & FLOPs
    params_info = count_torch_params(final_model)
    print(f"  Trainable params: {params_info['trainable_params']:,}, "
          f"Total params: {params_info['total_params']:,}")
    flops_info = estimate_torch_flops(final_model, (1, 1, 224, 224))
    print(f"  FLOPs (forward): {flops_info.get('total', 'N/A'):,}")

    # Save model
    model_path = RESULTS_DIR / "efficientnet_b0_model.pt"
    torch.save({
        "state_dict": final_model.state_dict(),
        "num_classes": num_classes,
        "strategy": "3-phase: feature-extraction → partial-unfreeze",
    }, model_path)
    print(f"  Model saved to {model_path}")

    # Plot loss / accuracy curves
    plot_loss_accuracy_curves(history, "EfficientNet-B0 (3-Phase)",
                              save_path=RESULTS_DIR / "efficientnet_b0_loss_accuracy.png")

    return {
        "model": "EfficientNet-B0 (3-Phase)",
        "test": test_metrics,
        "best_params": {
            "phase1_lr": 1e-3,
            "phase2_head_lr": 1e-4,
            "phase2_backbone_lr": 1e-5,
            "unfrozen_stages": 3,
            "early_stopping_patience": 8,
        },
        "param_info": params_info,
        "flops": flops_info,
        "history": history,
        "_fitted_model": final_model.cpu(),
    }


# ===================================================================
# K-Fold Cross-Validation evaluation  (with std over folds)
# ===================================================================
def run_kfold_cv(X_train, y_train, model_fn, model_name, n_folds=5, class_names=None):
    """Run Stratified K-Fold CV, reporting mean ± std for all metrics."""
    print(f"\n  --- {model_name} K-Fold CV (k={n_folds}) ---")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    fold_metrics: dict[str, list[float]] = defaultdict(list)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)

        # Get probabilities if available
        y_score = None
        if hasattr(model, "predict_proba"):
            try:
                y_score = model.predict_proba(X_va)
            except Exception:
                pass

        metrics = evaluate_model(y_va, y_pred, y_score, class_names)
        for k in ALL_METRIC_KEYS:
            if metrics.get(k) is not None:
                fold_metrics[k].append(metrics[k])

        # Always track F1 separately for backward compat
        print(f"    Fold {fold + 1}: F1 = {metrics['f1_macro']:.4f}, "
              f"TOP-1 = {metrics['top1_accuracy']:.4f}, "
              f"G-Mean = {metrics['g_mean']:.4f}")

    summary = {"model": model_name, "n_folds": n_folds}
    for k, vals in fold_metrics.items():
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary[f"{k}_scores"] = vals

    print(f"    Mean ± Std  →  "
          f"TOP-1: {summary.get('top1_accuracy_mean', 'N/A'):.4f} ± {summary.get('top1_accuracy_std', 0):.4f},  "
          f"F1: {summary.get('f1_macro_mean', 'N/A'):.4f} ± {summary.get('f1_macro_std', 0):.4f},  "
          f"G-Mean: {summary.get('g_mean_mean', 'N/A'):.4f} ± {summary.get('g_mean_std', 0):.4f}")
    if "roc_auc_macro_mean" in summary:
        print(f"    ROC-AUC: {summary['roc_auc_macro_mean']:.4f} ± {summary['roc_auc_macro_std']:.4f}")
    return summary


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate classifiers on spectrogram data.",
    )
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--models", type=str, nargs="+",
                        default=["svm", "rf", "xgb", "lda", "mlp", "transfer", "efficientnet"])
    parser.add_argument("--use-cv", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument("--pca-components", type=float, default=0.95)
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-cv", action="store_true")
    parser.add_argument("--no-tsne", action="store_true",
                        help="Skip t-SNE visualization.")
    args = parser.parse_args()

    global SEED
    SEED = args.seed

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading split dataset...")
    print("=" * 60)
    data = load_split_data(data_dir)
    class_names = data.get("metadata", {}).get(
        "class_names", ["DCBias", "Harmonic", "Loosen", "PartialDischarge"],
    )

    X_train_flat = data["X_train_flat"]
    X_val_flat = data["X_val_flat"]
    X_test_flat = data["X_test_flat"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    X_train_img = data["X_train_img"]
    X_val_img = data["X_val_img"]
    X_test_img = data["X_test_img"]

    print(f"Train: {X_train_flat.shape[0]}, Val: {X_val_flat.shape[0]}, Test: {X_test_flat.shape[0]}")
    print(f"Classes: {class_names}")
    print(f"Device: {DEVICE}")

    # ---- Preprocess ----
    print("\n" + "=" * 60)
    print("Preprocessing...")
    print("=" * 60)
    X_train_pp, X_val_pp, X_test_pp, scaler, pca = preprocess_flat(
        X_train_flat, X_val_flat, X_test_flat,
        use_pca=not args.no_pca,
        n_components=args.pca_components,
    )
    print(f"  Input dim:  {X_train_flat.shape[1]} → {X_train_pp.shape[1]} (after preprocessing)")

    # ---- t-SNE ----
    if not args.no_tsne and not args.only_cv:
        print("\n" + "=" * 60)
        print("t-SNE Visualization")
        print("=" * 60)
        X_all_pp = np.concatenate([X_train_pp, X_val_pp, X_test_pp], axis=0)
        y_all = np.concatenate([y_train, y_val, y_test], axis=0)
        plot_tsne(X_all_pp, y_all, class_names,
                  title="t-SNE of Preprocessed Spectrogram Features")

    # ---- K-Fold CV (standalone) ----
    if args.only_cv:
        print("\n" + "=" * 60)
        print(f"Running {args.cv_folds}-Fold CV on (train+val) ...")
        print("=" * 60)
        X_cv = np.concatenate([X_train_pp, X_val_pp])
        y_cv = np.concatenate([y_train, y_val])

        cv_results = []
        model_specs = {
            "svm": (lambda: SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                                random_state=SEED, class_weight="balanced"), "SVM (RBF)"),
            "rf": (lambda: RandomForestClassifier(n_estimators=200, max_depth=10,
                                                   random_state=SEED, class_weight="balanced"),
                   "Random Forest"),
            "xgb": (lambda: XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                           random_state=SEED, eval_metric="mlogloss", verbosity=0),
                    "XGBoost") if HAS_XGBOOST else (None, None),
            "lda": (lambda: LinearDiscriminantAnalysis(solver="svd"), "LDA"),
        }
        for mname in args.models:
            if mname in model_specs:
                fn, label = model_specs[mname]
                if fn is None:
                    continue
                cv_results.append(run_kfold_cv(
                    X_cv, y_cv, fn, label,
                    n_folds=args.cv_folds, class_names=class_names,
                ))

        if cv_results:
            cv_path = results_dir / "cross_validation_results.json"
            with open(cv_path, "w") as f:
                json.dump(cv_results, f, indent=2)
            print(f"\nCV results saved to {cv_path}")
            print("\n" + "-" * 60)
            print("Cross-Validation Summary:")
            print("-" * 60)
            for r in cv_results:
                print(f"  {r['model']:20s}: "
                      f"TOP-1 = {r.get('top1_accuracy_mean', 0):.4f} ± {r.get('top1_accuracy_std', 0):.4f}, "
                      f"F1 = {r.get('f1_macro_mean', 0):.4f} ± {r.get('f1_macro_std', 0):.4f}, "
                      f"G-Mean = {r.get('g_mean_mean', 0):.4f} ± {r.get('g_mean_std', 0):.4f}")
        print("\nDone (CV-only mode).")
        return

    # ---- Train models ----
    all_results = []
    model_registry = {
        "svm": (train_svm, True),
        "rf": (train_random_forest, True),
        "xgb": (train_xgboost, True),
        "lda": (train_lda, True),
        "mlp": (train_mlp, False),
        "transfer": (train_transfer_learning, False),
        "efficientnet": (train_efficientnet, False),
    }
    if "all" in args.models:
        args.models = list(model_registry.keys())

    for model_name in args.models:
        fn, _ = model_registry.get(model_name, (None, None))
        if fn is None:
            print(f"\n[WARNING] Unknown model: {model_name}. Skipping.")
            continue
        if model_name in ("transfer", "efficientnet") and args.skip_transfer:
            print(f"\n[SKIP] {model_name} skipped (--skip-transfer).")
            continue

        t0 = time.time()
        if model_name == "mlp":
            result = fn(X_train_pp, y_train, X_val_pp, y_val, X_test_pp, y_test,
                        class_names, use_cv=args.use_cv, cv_folds=args.cv_folds)
        elif model_name in ("transfer", "efficientnet"):
            result = fn(X_train_img, y_train, X_val_img, y_val, X_test_img, y_test,
                        class_names)
        else:
            result = fn(X_train_pp, y_train, X_val_pp, y_val, X_test_pp, y_test,
                        class_names, use_cv=args.use_cv, cv_folds=args.cv_folds)

        elapsed = time.time() - t0
        if result is not None:
            result["train_time_sec"] = elapsed
            print(f"\n  Training time: {elapsed:.1f}s")
            all_results.append(result)

            # Save individual report (strip non-serializable _fitted_model)
            report_path = results_dir / f"{model_name}_report.json"
            report_data = {k: v for k, v in result.items() if k != "_fitted_model"}
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2, default=str)

    # ---- Comparison summary ----
    print("\n" + "=" * 60)
    print("Model Comparison (Test Set)")
    print("=" * 60)

    valid_results = [r for r in all_results if r is not None]
    if valid_results:
        # Build dynamic header from available metrics
        metric_display = [
            ("top1_accuracy", "TOP-1"),
            ("precision_macro", "Precision"),
            ("recall_macro", "Recall"),
            ("f1_macro", "F1"),
            ("g_mean", "G-Mean"),
            ("roc_auc_macro", "ROC-AUC"),
        ]
        available_cols = [(k, d) for k, d in metric_display
                          if all(k in r["test"] and r["test"][k] is not None for r in valid_results)]

        header_parts = [f"{'Model':<30s}"] + [f"{d:>10s}" for _, d in available_cols]
        header = "".join(header_parts)
        print(header)
        print("-" * len(header))

        for r in valid_results:
            t = r["test"]
            parts = [f"{r['model']:<30s}"] + [
                f"{t[k]:>10.4f}" if t.get(k) is not None else f"{'N/A':>10s}"
                for k, _ in available_cols
            ]
            print("".join(parts))

        # Also print param / FLOPs summary
        print("\n" + "-" * 60)
        print(f"{'Model':<30s} {'Params':>20s} {'FLOPs':>20s}")
        print("-" * 72)
        for r in valid_results:
            pi = r.get("param_info", {})
            fl = r.get("flops", {})
            if "trainable_params" in pi:
                p_str = f"{pi['trainable_params']:,} / {pi.get('total_params', '?'):,}"
            elif "effective_params" in pi:
                p_str = str(pi["effective_params"])
            else:
                p_str = "N/A"
            fl_str = f"{fl.get('total', 'N/A'):,}" if isinstance(fl, dict) else str(fl)
            print(f"{r['model']:<30s} {p_str:>20s} {fl_str:>20s}")

        # Save CSV with all metrics
        rows = []
        for r in valid_results:
            t = r["test"]
            row = {"model": r["model"]}
            for k in ["top1_accuracy", "precision_macro", "recall_macro", "f1_macro",
                       "g_mean", "roc_auc_macro", "roc_auc_weighted"]:
                row[k] = t.get(k)
            row["train_time_sec"] = r.get("train_time_sec", 0)
            row["best_params"] = str(r.get("best_params", {}))
            # Params / FLOPs
            pi = r.get("param_info", {})
            fl = r.get("flops", {})
            row["trainable_params"] = pi.get("trainable_params", pi.get("effective_params", "N/A"))
            row["total_params"] = pi.get("total_params", "N/A")
            row["flops_forward"] = fl.get("total", "N/A") if isinstance(fl, dict) else str(fl)
            rows.append(row)

        csv_path = results_dir / "model_comparison.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nComparison CSV saved to {csv_path}")

        json_path = results_dir / "model_comparison.json"
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Comparison JSON saved to {json_path}")

        # Plots
        plot_confusion_matrices(valid_results, class_names)
        plot_model_comparison(valid_results)

        # Per-model t-SNE grid (learned embeddings & input-space views)
        plot_tsne_per_model(valid_results, X_test_pp, X_test_img,
                           y_test, class_names, results_dir)
    else:
        print("No models were successfully trained.")

    print("\nDone.")


if __name__ == "__main__":
    main()
