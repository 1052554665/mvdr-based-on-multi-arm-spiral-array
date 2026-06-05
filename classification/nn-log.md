>how does the t-sne work? could the t-sne reflect the classification ability of  different models? could plot the t-sne for all models?


## Two t-SNE outputs

|Output  | Source | What it shows|
|-------|------|---------------|
| `tsne_embedding.png`	 | Raw preprocessed input features |Natural separability of the input space|
| `tsne_per_model.png`	 | Per-model feature representations	 | How each model "sees" the data|

## Key design decisions
- **Models with learned embeddings** (LDA, MLP, CNNs): t-SNE of the learned feature space colored by true labels — reveals whether the model has structured its internal representation to separate classes.
- **Models without** (SVM, RF, XGBoost): t-SNE of the input space colored by true labels — shows what the classifier sees, since these models operate directly on the input.
- Each model's fitted object is stored in result `["_fitted_model"]` and stripped before JSON serialization.