#!/usr/bin/env python3
"""
Split the spectrogram dataset into training, validation, and test sets
with stratified splitting (70/15/15) and Stratified K-Fold cross-validation.

Usage:
    cd classification
    python split_dataset.py                          # default: 70/15/15 with 5-fold CV
    python split_dataset.py --k-folds 10             # 10-fold CV
    python split_dataset.py --no-cv                  # skip K-Fold generation
    python split_dataset.py --use-noisy              # use noisy spectrogram instead of enhanced

Data source:  transformer/datasets-npz/{DCBias, Harmonic, Loosen, PartialDischarge}/*.npz
Output dir:   classification/data/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["DCBias", "Harmonic", "Loosen", "PartialDischarge"]
DEFAULT_DATA_KEY = "enhanced"  # key inside each .npz to use as the feature
EXPECTED_SHAPE = (1, 128, 64)   # single-channel mel-spectrogram


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(dataset_dir: Path, data_key: str = DEFAULT_DATA_KEY):
    """Load all .npz files from the class-named subdirectories.

    Parameters
    ----------
    dataset_dir : Path
        Root directory containing one subdirectory per class.
    data_key : str
        Which array inside the .npz to extract (``"enhanced"``, ``"mask"``,
        or ``"noisy"``).

    Returns
    -------
    X_flat : np.ndarray  shape (n_samples, 8192)
        Flattened spectrograms for traditional ML models.
    X_img : np.ndarray   shape (n_samples, 1, 128, 64)
        Image-shaped spectrograms for CNN / transfer learning.
    y : np.ndarray       shape (n_samples,)
        Integer labels 0-3.
    class_names : list[str]
        Ordered class names (same order as label indices).
    file_paths : list[str]
        Absolute path of each loaded file (for traceability).
    """
    X_flat_list = []
    X_img_list = []
    y_list = []
    file_paths = []

    # Sort for deterministic ordering
    class_dirs = sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    class_names = [d.name for d in class_dirs]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    print(f"Found {len(class_dirs)} class directories: {class_names}")

    for class_name in class_names:
        class_dir = dataset_dir / class_name
        npz_files = sorted(class_dir.glob("*.npz"))
        print(f"  {class_name}: {len(npz_files)} files")

        for npz_file in npz_files:
            data = np.load(str(npz_file))
            if data_key not in data:
                print(f"  [WARNING] key '{data_key}' not found in {npz_file}, skipping.")
                data.close()
                continue

            spec = data[data_key]  # shape varies by key

            # Handle different shapes
            if spec.ndim == 3 and spec.shape[0] == 1:
                # (1, H, W) → keep as-is for images, flatten for tabular
                img = spec.astype(np.float32)
            elif spec.ndim == 3 and spec.shape[0] == 2:
                # (2, H, W) → average to single channel
                img = spec.mean(axis=0, keepdims=True).astype(np.float32)
            elif spec.ndim == 2:
                img = spec[np.newaxis, :, :].astype(np.float32)
            else:
                print(f"  [WARNING] unexpected shape {spec.shape} in {npz_file}, skipping.")
                data.close()
                continue

            X_img_list.append(img)
            X_flat_list.append(img.flatten())
            y_list.append(class_to_idx[class_name])
            file_paths.append(str(npz_file.resolve()))
            data.close()

    X_flat = np.array(X_flat_list, dtype=np.float32)
    X_img = np.array(X_img_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"\nTotal samples: {len(y)}")
    for i, name in enumerate(class_names):
        print(f"  {name}: {(y == i).sum()}")

    return X_flat, X_img, y, class_names, file_paths


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def stratified_split(X, y, test_size=0.30, val_ratio=0.50, seed=42):
    """Split into train / val / test with stratification.

    Returns indices for each split.
    """
    n = len(y)
    indices = np.arange(n)

    # First split: train vs temp (val + test)
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, y, test_size=test_size, stratify=y, random_state=seed,
    )

    # Second split: val vs test from temp
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, test_size=val_ratio, stratify=y_temp, random_state=seed,
    )

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# K-Fold generation
# ---------------------------------------------------------------------------
def generate_kfold_splits(X_train, y_train, n_splits=5, seed=42):
    """Generate stratified K-Fold splits on the training set.

    Returns a list of dicts with ``train`` and ``val`` index arrays.
    """
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed,
    )
    folds = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        folds.append({"fold": fold_idx, "train": tr_idx, "val": va_idx})
    return folds


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_splits(
    output_dir: Path,
    X_flat, X_img, y,
    train_idx, val_idx, test_idx,
    class_names,
    file_paths,
):
    """Save flat features, image features, labels, and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Flat features (for traditional ML) --
    np.save(output_dir / "X_train_flat.npy", X_flat[train_idx])
    np.save(output_dir / "X_val_flat.npy", X_flat[val_idx])
    np.save(output_dir / "X_test_flat.npy", X_flat[test_idx])

    # -- Image features (for CNN / transfer learning) --
    np.save(output_dir / "X_train_img.npy", X_img[train_idx])
    np.save(output_dir / "X_val_img.npy", X_img[val_idx])
    np.save(output_dir / "X_test_img.npy", X_img[test_idx])

    # -- Labels --
    np.save(output_dir / "y_train.npy", y[train_idx])
    np.save(output_dir / "y_val.npy", y[val_idx])
    np.save(output_dir / "y_test.npy", y[test_idx])

    # -- File paths (for traceability) --
    fp_array = np.array(file_paths, dtype=object)
    np.save(output_dir / "file_paths_train.npy", fp_array[train_idx])
    np.save(output_dir / "file_paths_val.npy", fp_array[val_idx])
    np.save(output_dir / "file_paths_test.npy", fp_array[test_idx])

    # -- Metadata --
    metadata = {
        "class_names": class_names,
        "class_to_idx": {n: i for i, n in enumerate(class_names)},
        "num_classes": len(class_names),
        "total_samples": len(y),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "feature_shape_flat": X_flat.shape[1],
        "feature_shape_img": list(X_img.shape[1:]),
        "class_distribution": {
            "train": {class_names[i]: int((y[train_idx] == i).sum()) for i in range(len(class_names))},
            "val": {class_names[i]: int((y[val_idx] == i).sum()) for i in range(len(class_names))},
            "test": {class_names[i]: int((y[test_idx] == i).sum()) for i in range(len(class_names))},
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def save_kfold(output_dir: Path, folds, y_train):
    """Save K-Fold split indices."""
    kfold_dir = output_dir / "kfold"
    kfold_dir.mkdir(parents=True, exist_ok=True)

    for entry in folds:
        fold = entry["fold"]
        np.save(kfold_dir / f"fold{fold}_train_idx.npy", entry["train"])
        np.save(kfold_dir / f"fold{fold}_val_idx.npy", entry["val"])

    # Summary
    summary = {
        "n_folds": len(folds),
        "folds": [
            {
                "fold": e["fold"],
                "train_size": len(e["train"]),
                "val_size": len(e["val"]),
                "train_dist": {
                    CLASS_NAMES[i]: int((y_train[e["train"]] == i).sum())
                    for i in range(len(CLASS_NAMES))
                },
                "val_dist": {
                    CLASS_NAMES[i]: int((y_train[e["val"]] == i).sum())
                    for i in range(len(CLASS_NAMES))
                },
            }
            for e in folds
        ],
    }
    with open(kfold_dir / "kfold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {len(folds)}-Fold CV splits to {kfold_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Split spectrogram dataset into train/val/test with stratification.",
    )
    parser.add_argument(
        "--data-dir", type=str, default="transformer/datasets-npz",
        help="Root directory containing class subdirectories with .npz files.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="classification/data",
        help="Directory to save the split datasets.",
    )
    parser.add_argument(
        "--data-key", type=str, default=DEFAULT_DATA_KEY,
        choices=["enhanced", "mask", "noisy"],
        help="Which array inside each .npz to use as the feature.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.30,
        help="Proportion for (val + test) combined.",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.50,
        help="Proportion of the temp set to use as validation (rest becomes test).",
    )
    parser.add_argument(
        "--k-folds", type=int, default=5,
        help="Number of stratified K-Fold splits on the training set.",
    )
    parser.add_argument(
        "--no-cv", action="store_true",
        help="Skip K-Fold generation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    # Resolve paths relative to the project root (where this script lives = classification/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        sys.exit(1)

    # ---- Load ----
    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)
    X_flat, X_img, y, class_names, file_paths = load_dataset(data_dir, args.data_key)

    # ---- Split ----
    print("\n" + "=" * 60)
    print("Stratified 70/15/15 split...")
    print("=" * 60)
    train_idx, val_idx, test_idx = stratified_split(
        X_flat, y,
        test_size=args.test_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"Train:      {len(train_idx)} samples")
    for i, name in enumerate(class_names):
        print(f"  {name}: {(y[train_idx] == i).sum()}")
    print(f"Validation: {len(val_idx)} samples")
    for i, name in enumerate(class_names):
        print(f"  {name}: {(y[val_idx] == i).sum()}")
    print(f"Test:       {len(test_idx)} samples")
    for i, name in enumerate(class_names):
        print(f"  {name}: {(y[test_idx] == i).sum()}")

    # ---- Save splits ----
    print("\n" + "=" * 60)
    print(f"Saving split data to {output_dir} ...")
    print("=" * 60)
    metadata = save_splits(
        output_dir, X_flat, X_img, y,
        train_idx, val_idx, test_idx,
        class_names, file_paths,
    )

    print(f"\nSaved files to {output_dir}/:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name}  ({size_kb:.1f} KB)")

    # ---- K-Fold ----
    if not args.no_cv:
        print("\n" + "=" * 60)
        print(f"Generating {args.k_folds}-Fold stratified CV splits...")
        print("=" * 60)
        y_train = y[train_idx]
        folds = generate_kfold_splits(
            X_flat[train_idx], y_train,
            n_splits=args.k_folds,
            seed=args.seed,
        )
        save_kfold(output_dir, folds, y_train)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Classes:           {class_names}")
    print(f"Total samples:     {metadata['total_samples']}")
    print(f"Feature dim (flat): {metadata['feature_shape_flat']}")
    print(f"Feature dim (img):  {metadata['feature_shape_img']}")
    print(f"Train / Val / Test: {metadata['train_samples']} / {metadata['val_samples']} / {metadata['test_samples']}")
    print("\nClass distribution:")
    for split in ["train", "val", "test"]:
        dist = metadata["class_distribution"][split]
        parts = [f"{cls}={dist[cls]}" for cls in class_names]
        print(f"  {split:5s}: {', '.join(parts)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
