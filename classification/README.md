# step 1: Split Dataset

`split_dataset.py`

Split the spectrogram dataset into training, validation, and test sets
with stratified splitting (70/15/15) and Stratified K-Fold cross-validation.


Usage:
```bash
    cd classification
    python split_dataset.py                          # default: 70/15/15 with 5-fold CV
    python split_dataset.py --k-folds 10             # 10-fold CV
    python split_dataset.py --no-cv                  # skip K-Fold generation
    python split_dataset.py --use-noisy              # use noisy spectrogram instead of enhanced
```

- Data source:  `transformer/datasets-npz/{DCBias, Harmonic, Loosen, PartialDischarge}/*.npz`

- Output dir:   `classification/data/`


# step 2: Train Neural Networks

```bash
cd classification
python train_classifier.py                  # train all models
python train_classifier.py --models svm rf  # train only SVM + Random Forest
python train_classifier.py --use-cv         # use K-Fold CV instead of fixed val split
python train_classifier.py --no-pca         # disable PCA dimensionality reduction
```

Train and evaluate multiple classifiers on the split spectrogram dataset.

Models are trained in order of recommendation priority for small datasets:

    1. SVM (RBF kernel)        — often best for small, structured data
    2. Random Forest            — robust, handles noise well
    3. XGBoost                  — great for tabular data
    4. LDA                      — great if classes are linearly separable
    5. MLP (small)              — with strong regularization
    6. Transfer Learning (CNN)  — treats spectrograms as images

Metrics computed:

    TOP-1 Accuracy, Precision, Recall, F1-score, G-Mean,
    ROC-AUC (OVR), Confusion Matrix, Parameter count, FLOPs (PyTorch models)

Visualizations:

    t-SNE embedding, Confusion matrices, Loss / Accuracy curves,
    Model comparison bar chart


Usage:
```bash
    cd classification
    python split_dataset.py                    # run this first
    python train_classifier.py                  # train all models
    python train_classifier.py --models svm rf  # train only SVM + Random Forest
    python train_classifier.py --use-cv         # use K-Fold CV instead of fixed val split
    python train_classifier.py --no-pca         # disable PCA dimensionality reduction
```

Output:
```plaintext
    classification/results/
    ├── model_comparison.csv
    ├── model_comparison.json
    ├── confusion_matrices.png
    ├── tsne_embedding.png
    ├── model_comparison.png
    ├── <model>_loss_accuracy.png      (MLP / Transfer Learning)
    ├── svm_report.json
    └── ...
```


