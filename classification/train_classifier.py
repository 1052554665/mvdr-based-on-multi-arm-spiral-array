#!/usr/bin/env python3
"""
Train and evaluate multiple classifiers on the split spectrogram dataset.

Models are trained in order of recommendation priority for small datasets:

    1. SVM (RBF kernel)        — often best for small, structured data
    2. Random Forest            — robust, handles noise well
    3. XGBoost                  — great for tabular data
    4. LDA                      — great if classes are linearly separable
    5. MLP (small)              — with strong regularization
    6. Transfer Learning (CNN)  — treats spectrograms as images

Usage:
    cd classification
    python split_dataset.py                    # run this first
    python train_classifier.py                  # train all models
    python train_classifier.py --models svm rf  # train only SVM + Random Forest
    python train_classifier.py --use-cv         # use K-Fold CV instead of fixed val split
    python train_classifier.py --no-pca         # disable PCA dimensionality reduction

Output:
    classification/results/
    ├── model_comparison.csv       # side-by-side metrics
    ├── confusion_matrices.png     # 2×3 grid of confusion matrices
    ├── svm_report.json
    ├── random_forest_report.json
    └── ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Scikit-learn models
# ---------------------------------------------------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
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
    from torchvision import transforms as T

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
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


# ===================================================================
# Data loading utilities
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

    # Load metadata
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
        n_folds = summary["n_folds"]
        for fi in range(n_folds):
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
        from sklearn.decomposition import PCA

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
# Evaluation helpers
# ===================================================================
def evaluate_model(y_true, y_pred, class_names):
    """Return a dict of metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_per_class": precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0,
        ),
    }


def print_metrics(name, metrics):
    """Pretty-print evaluation metrics."""
    print(f"\n  --- {name} ---")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision_macro']:.4f} (macro)")
    print(f"  Recall:    {metrics['recall_macro']:.4f} (macro)")
    print(f"  F1:        {metrics['f1_macro']:.4f} (macro)")
    if "classification_report" in metrics:
        print(f"\n{metrics['classification_report']}")


# ===================================================================
# 1. SVM (RBF kernel)
# ===================================================================
def train_svm(X_train, y_train, X_val, y_val, X_test, y_test, class_names, use_cv=False, cv_folds=5):
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
        y_pred_test = grid.predict(X_test)
        test_metrics = evaluate_model(y_test, y_pred_test, class_names)
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
        val_metrics = evaluate_model(y_val, y_pred_val, class_names)
        print(f"  Best params: {grid.best_params_}")
        print_metrics("Validation", val_metrics)

        y_pred_test = best_model.predict(X_test)
        test_metrics = evaluate_model(y_test, y_pred_test, class_names)

    print_metrics("Test", test_metrics)
    return {"model": "SVM (RBF)", "test": test_metrics, "best_params": grid.best_params_}


# ===================================================================
# 2. Random Forest
# ===================================================================
def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, class_names, use_cv=False, cv_folds=5):
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
        y_pred_test = grid.predict(X_test)
        test_metrics = evaluate_model(y_test, y_pred_test, class_names)
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
        val_metrics = evaluate_model(y_val, y_pred_val, class_names)
        print(f"  Best params: {grid.best_params_}")
        print_metrics("Validation", val_metrics)

        y_pred_test = best_model.predict(X_test)
        test_metrics = evaluate_model(y_test, y_pred_test, class_names)

    print_metrics("Test", test_metrics)

    # Feature importance
    importances = grid.best_estimator_.feature_importances_
    top_n = min(10, len(importances))
    top_idx = np.argsort(importances)[::-1][:top_n]
    print(f"\n  Top {top_n} feature importances:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"    {rank}. Feature {idx}: {importances[idx]:.4f}")

    return {"model": "Random Forest", "test": test_metrics, "best_params": grid.best_params_}


# ===================================================================
# 3. XGBoost
# ===================================================================
def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, class_names, use_cv=False, cv_folds=5):
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
        y_pred_test = grid.predict(X_test)
        test_metrics = evaluate_model(y_test, y_pred_test, class_names)
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
        val_metrics = evaluate_model(y_val, y_pred_val, class_names)
        print(f"  Best params: {grid.best_params_}")
        print_metrics("Validation", val_metrics)

        y_pred_test = best_model.predict(X_test)
        test_metrics = evaluate_model(y_test, y_pred_test, class_names)

    print_metrics("Test", test_metrics)
    return {"model": "XGBoost", "test": test_metrics, "best_params": grid.best_params_}


# ===================================================================
# 4. LDA
# ===================================================================
def train_lda(X_train, y_train, X_val, y_val, X_test, y_test, class_names, use_cv=False, cv_folds=5):
    print("\n" + "=" * 60)
    print("4. Linear Discriminant Analysis (LDA)")
    print("=" * 60)

    # LDA needs n_components <= n_classes - 1
    n_components_max = len(class_names) - 1

    if use_cv:
        # Try different shrinkage values via CV
        from sklearn.model_selection import cross_val_score

        X_all = np.concatenate([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])

        best_score = -1
        best_shrinkage = None
        for shrinkage in [None, "auto", 0.1, 0.5, 1.0]:
            try:
                lda = LinearDiscriminantAnalysis(
                    n_components=n_components_max, shrinkage=shrinkage, solver="eigen" if shrinkage else "svd",
                )
                scores = cross_val_score(lda, X_all, y_all, cv=cv_folds, scoring="f1_macro")
                mean_score = scores.mean()
                if mean_score > best_score:
                    best_score = mean_score
                    best_shrinkage = shrinkage
            except Exception:
                continue

        lda = LinearDiscriminantAnalysis(
            n_components=n_components_max, shrinkage=best_shrinkage,
            solver="eigen" if best_shrinkage else "svd",
        )
        lda.fit(X_all, y_all)
        y_pred_test = lda.predict(X_test)
        test_metrics = evaluate_model(y_test, y_pred_test, class_names)
        print(f"  Shrinkage: {best_shrinkage}")
    else:
        # Try a few shrinkage values, pick best on val
        best_val_f1 = -1
        best_lda = None
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
        val_metrics = evaluate_model(y_val, y_pred_val, class_names)
        print(f"  Shrinkage: {best_shrinkage}")
        print_metrics("Validation", val_metrics)

        y_pred_test = best_lda.predict(X_test)
        test_metrics = evaluate_model(y_test, y_pred_test, class_names)

    print_metrics("Test", test_metrics)
    return {"model": "LDA", "test": test_metrics, "best_params": {"shrinkage": best_shrinkage}}


# ===================================================================
# 5. MLP (small, PyTorch)
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


def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, class_names, use_cv=False, cv_folds=5):
    print("\n" + "=" * 60)
    print("5. MLP (Small, Regularized)")
    print("=" * 60)

    if not HAS_TORCH:
        print("  [SKIP] PyTorch not installed.")
        return None

    input_dim = X_train.shape[1]
    num_classes = len(class_names)

    # Convert to tensors
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

    # Hyperparameter search (simple grid)
    configs = [
        {"hidden_dims": (64, 32), "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-3},
        {"hidden_dims": (128, 64), "dropout": 0.5, "lr": 1e-3, "weight_decay": 1e-3},
        {"hidden_dims": (64,), "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-2},
        {"hidden_dims": (128, 64, 32), "dropout": 0.5, "lr": 1e-4, "weight_decay": 1e-2},
    ]

    best_val_f1 = -1
    best_model_state = None
    best_config = None

    for cfg in configs:
        torch.manual_seed(SEED)
        model = SmallMLP(input_dim, num_classes, cfg["hidden_dims"], cfg["dropout"]).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        criterion = nn.CrossEntropyLoss()

        # Train
        model.train()
        for epoch in range(100):
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            scheduler.step()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        print(f"  Config {cfg}: val_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_config = cfg

    # Retrain best config on train+val combined for final eval
    print(f"\n  Best config: {best_config}, val_f1={best_val_f1:.4f}")

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

    final_model.train()
    for epoch in range(150):
        total_loss = 0.0
        for xb, yb in combined_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = final_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        scheduler.step()

    # Test
    final_model.eval()
    with torch.no_grad():
        logits = final_model(X_test_t.to(DEVICE))
        y_pred_test = logits.argmax(dim=1).cpu().numpy()

    test_metrics = evaluate_model(y_test, y_pred_test, class_names)
    print_metrics("Test", test_metrics)

    # Save model
    model_path = RESULTS_DIR / "mlp_model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": final_model.state_dict(), "config": best_config,
                "input_dim": input_dim, "num_classes": num_classes}, model_path)
    print(f"  Model saved to {model_path}")

    return {"model": "MLP (Small)", "test": test_metrics, "best_params": best_config}


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
            # Replace first conv to accept 1 channel instead of 3
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False,
            )
            # Initialize the new conv by averaging the pretrained RGB weights
            with torch.no_grad():
                self.backbone.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze layer4 and the new conv1
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


def train_transfer_learning(X_train_img, y_train, X_val_img, y_val, X_test_img, y_test, class_names):
    print("\n" + "=" * 60)
    print("6. Transfer Learning (ResNet18)")
    print("=" * 60)

    if not HAS_TORCH or not HAS_TORCHVISION:
        print("  [SKIP] PyTorch / torchvision not installed.")
        return None

    num_classes = len(class_names)

    # The input shape is (N, 1, 128, 64). ResNet needs 224×224.
    # We resize on-the-fly using torch interpolation.
    def resize_batch(batch):
        """Resize (N, 1, H, W) to (N, 1, 224, 224)."""
        return nn.functional.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)

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

    # Try both frozen and fine-tuned
    configs = [
        {"freeze": True, "lr": 1e-3, "label": "frozen_backbone"},
        {"freeze": False, "lr": 1e-4, "label": "fine_tuned"},
    ]

    best_val_f1 = -1
    best_model_state = None
    best_config_label = None

    for cfg in configs:
        torch.manual_seed(SEED)
        model = SpectrogramClassifier(num_classes, freeze_backbone=cfg["freeze"]).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(50):
            total_loss = 0.0
            for xb, yb in train_loader:
                xb = resize_batch(xb.to(DEVICE))
                yb = yb.to(DEVICE)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            scheduler.step()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = resize_batch(xb.to(DEVICE))
                logits = model(xb)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        print(f"  {cfg['label']}: val_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_config_label = cfg["label"]

    if best_model_state is None:
        print("  [ERROR] Transfer learning failed.")
        return None

    # Retrain best config on train+val
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

    final_model.train()
    for epoch in range(80):
        total_loss = 0.0
        for xb, yb in combined_loader:
            xb = resize_batch(xb.to(DEVICE))
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = final_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        scheduler.step()

    # Test
    final_model.eval()
    with torch.no_grad():
        logits = final_model(resize_batch(X_test_t.to(DEVICE)))
        y_pred_test = logits.argmax(dim=1).cpu().numpy()

    test_metrics = evaluate_model(y_test, y_pred_test, class_names)
    print_metrics("Test", test_metrics)

    # Save model
    model_path = RESULTS_DIR / "transfer_learning_model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": final_model.state_dict(), "config": best_config_label,
                "num_classes": num_classes}, model_path)
    print(f"  Model saved to {model_path}")

    return {"model": "Transfer Learning (ResNet18)", "test": test_metrics,
            "best_params": {"freeze_backbone": best_freeze}}


# ===================================================================
# Plotting
# ===================================================================
def plot_confusion_matrices(all_results, class_names):
    """Plot a grid of confusion matrices for all trained models."""
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
        cm_norm = np.array(cm).astype(float)
        cm_norm = cm_norm / cm_norm.sum(axis=1, keepdims=True)  # row-normalize
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0, vmax=1, ax=ax, cbar=(idx == n - 1))
        ax.set_title(model_name, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    # Hide unused subplots
    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].set_visible(False)

    fig.suptitle("Confusion Matrices (Row-Normalized)", fontsize=14, fontweight="bold")
    fig.tight_layout()

    plot_path = RESULTS_DIR / "confusion_matrices.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nConfusion matrices saved to {plot_path}")


def plot_comparison(all_results):
    """Bar chart comparing test metrics across models."""
    if not HAS_PLOTTING:
        return

    valid = [r for r in all_results if r is not None]
    if not valid:
        return

    models = [r["model"] for r in valid]
    metrics_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, mname in enumerate(metrics_names):
        values = [r["test"][mname] for r in valid]
        bars = ax.bar(x + i * width, values, width, label=mname.replace("_macro", " (macro)").replace("_", " ").title())
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
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
# K-Fold Cross-Validation evaluation
# ===================================================================
def run_kfold_cv(X_train, y_train, model_fn, model_name, n_folds=5):
    """Run K-Fold CV for a given model factory function."""
    print(f"\n  --- {model_name} K-Fold CV (k={n_folds}) ---")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        f1 = f1_score(y_va, y_pred, average="macro", zero_division=0)
        fold_scores.append(f1)
        print(f"    Fold {fold + 1}: F1 = {f1:.4f}")

    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores))
    print(f"    Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
    return {"model": model_name, "cv_f1_mean": mean_f1, "cv_f1_std": std_f1, "cv_f1_scores": fold_scores}


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate classifiers on spectrogram data.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA_DIR),
        help="Directory containing the split dataset files.",
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(RESULTS_DIR),
        help="Directory to save results, model files, and plots.",
    )
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=["svm", "rf", "xgb", "lda", "mlp", "transfer"],
        help="Which models to train. Options: svm, rf, xgb, lda, mlp, transfer, all.",
    )
    parser.add_argument(
        "--use-cv", action="store_true",
        help="Use K-Fold CV instead of fixed validation split for sklearn models.",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of folds for cross-validation.",
    )
    parser.add_argument(
        "--no-pca", action="store_true",
        help="Disable PCA dimensionality reduction.",
    )
    parser.add_argument(
        "--pca-components", type=float, default=0.95,
        help="Number of PCA components (int) or variance ratio (float).",
    )
    parser.add_argument(
        "--skip-transfer", action="store_true",
        help="Skip transfer learning (requires GPU for reasonable speed).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--only-cv", action="store_true",
        help="Only run K-Fold CV evaluation (skip separate val split training).",
    )
    args = parser.parse_args()

    global SEED
    SEED = args.seed

    # ---- Load data ----
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading split dataset...")
    print("=" * 60)
    data = load_split_data(data_dir)
    class_names = data.get("metadata", {}).get("class_names", ["DCBias", "Harmonic", "Loosen", "PartialDischarge"])

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

    # ---- K-Fold CV (optional standalone) ----
    if args.only_cv:
        print("\n" + "=" * 60)
        print(f"Running {args.cv_folds}-Fold CV on training set...")
        print("=" * 60)

        X_cv = np.concatenate([X_train_pp, X_val_pp])
        y_cv = np.concatenate([y_train, y_val])

        cv_results = []
        for model_name in args.models:
            if model_name in ("svm", "all"):
                cv_results.append(run_kfold_cv(
                    X_cv, y_cv,
                    lambda: SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=SEED),
                    "SVM (RBF)", n_folds=args.cv_folds,
                ))
            if model_name in ("rf", "all"):
                cv_results.append(run_kfold_cv(
                    X_cv, y_cv,
                    lambda: RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED),
                    "Random Forest", n_folds=args.cv_folds,
                ))
            if model_name in ("xgb", "all") and HAS_XGBOOST:
                cv_results.append(run_kfold_cv(
                    X_cv, y_cv,
                    lambda: XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                          random_state=SEED, eval_metric="mlogloss", verbosity=0),
                    "XGBoost", n_folds=args.cv_folds,
                ))
            if model_name in ("lda", "all"):
                cv_results.append(run_kfold_cv(
                    X_cv, y_cv,
                    lambda: LinearDiscriminantAnalysis(solver="svd"),
                    "LDA", n_folds=args.cv_folds,
                ))

        if cv_results:
            cv_results = [r for r in cv_results if r is not None]
            cv_path = results_dir / "cross_validation_results.json"
            with open(cv_path, "w") as f:
                json.dump(cv_results, f, indent=2)
            print(f"\nCV results saved to {cv_path}")

            # Print summary
            print("\n" + "-" * 40)
            print("Cross-Validation Summary:")
            print("-" * 40)
            for r in cv_results:
                print(f"  {r['model']:20s}: F1 = {r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}")

        print("\nDone (CV-only mode).")
        return

    # ---- Train models ----
    all_results = []
    model_registry = {
        "svm": (train_svm, True),
        "rf": (train_random_forest, True),
        "xgb": (train_xgboost, True),
        "lda": (train_lda, True),
        "mlp": (train_mlp, False),  # uses its own preprocessing
        "transfer": (train_transfer_learning, False),  # uses image data
    }

    if "all" in args.models:
        args.models = list(model_registry.keys())

    # Train sklearn models (use preprocessed flat data)
    for model_name in args.models:
        fn, uses_flat = model_registry.get(model_name, (None, None))
        if fn is None:
            print(f"\n[WARNING] Unknown model: {model_name}. Skipping.")
            continue

        if model_name == "transfer" and args.skip_transfer:
            print("\n[SKIP] Transfer learning skipped (--skip-transfer).")
            continue

        t0 = time.time()

        if model_name == "mlp":
            result = fn(X_train_pp, y_train, X_val_pp, y_val, X_test_pp, y_test, class_names,
                        use_cv=args.use_cv, cv_folds=args.cv_folds)
        elif model_name == "transfer":
            result = fn(X_train_img, y_train, X_val_img, y_val, X_test_img, y_test, class_names)
        else:
            result = fn(X_train_pp, y_train, X_val_pp, y_val, X_test_pp, y_test, class_names,
                        use_cv=args.use_cv, cv_folds=args.cv_folds)

        elapsed = time.time() - t0
        if result is not None:
            result["train_time_sec"] = elapsed
            print(f"\n  Training time: {elapsed:.1f}s")
            all_results.append(result)

            # Save individual report
            report_path = results_dir / f"{model_name}_report.json"
            # Convert non-serializable items
            report_data = {k: v for k, v in result.items()}
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2, default=str)

    # ---- Comparison summary ----
    print("\n" + "=" * 60)
    print("Model Comparison (Test Set)")
    print("=" * 60)

    valid_results = [r for r in all_results if r is not None]
    if valid_results:
        header = f"{'Model':<30s} {'Accuracy':>8s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}"
        print(header)
        print("-" * len(header))
        for r in valid_results:
            t = r["test"]
            print(f"{r['model']:<30s} {t['accuracy']:>8.4f} {t['precision_macro']:>10.4f} "
                  f"{t['recall_macro']:>10.4f} {t['f1_macro']:>10.4f}")

        # Save CSV
        rows = []
        for r in valid_results:
            t = r["test"]
            rows.append({
                "model": r["model"],
                "accuracy": t["accuracy"],
                "precision_macro": t["precision_macro"],
                "recall_macro": t["recall_macro"],
                "f1_macro": t["f1_macro"],
                "train_time_sec": r.get("train_time_sec", 0),
                "best_params": str(r.get("best_params", {})),
            })

        import csv
        csv_path = results_dir / "model_comparison.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nComparison CSV saved to {csv_path}")

        # Save as JSON too
        json_path = results_dir / "model_comparison.json"
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Comparison JSON saved to {json_path}")

        # Plots
        if args.models != ["transfer"]:  # transfer uses different input, cm shape same though
            plot_confusion_matrices(valid_results, class_names)
        plot_comparison(valid_results)
    else:
        print("No models were successfully trained.")

    print("\nDone.")


if __name__ == "__main__":
    main()
