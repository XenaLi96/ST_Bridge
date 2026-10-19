# Evaluation definitions

`evaluate` reads saved predictions; it does not modify, gate, rescale, or calibrate them with reference expression. The default split is held-out **spots**. `all` must be explicitly selected.

- `flattened_regression`: flatten cells × genes, then compute MAE, MSE, RMSE, Pearson, Spearman, R², and cosine similarity.
- `mean_gene_metrics`: compute each metric separately over cells for each gene, then average finite gene-level values with equal weight per gene.
- `pooled_activation`: compute activation statistics over all selected cell–gene entries.
- The per-gene CSV contains regression, activation, one-dimensional energy distance, and biased squared MMD-RBF estimates.

Pearson/Spearman require at least two entries and variability in both vectors. R² requires variability in the reference vector. Undefined correlations/R²/cosine are JSON `null` and empty CSV cells; finite means omit those values. Undefined is not silently converted to a perfect score. Cosine is the vector dot product divided by the two L2 norms.

Activation is strictly `expression > threshold`, default 0.3 on the example normalized expression scale. Precision and recall use zero when their denominator is zero. F1 is `2TP/(2TP+FP+FN)`, zero when its denominator is zero. Accuracy is mean binary agreement. Rate difference is the absolute difference between reference and predicted active fractions. Thresholds are caller settings; they are not fitted on test expression.

Energy distance uses `scipy.stats.energy_distance` on each gene's one-dimensional cell-expression samples. MMD is the biased squared estimate `mean(k(x,x)) + mean(k(y,y)) - 2mean(k(x,y))`, with `k(a,b)=exp(-(a-b)^2/(2*bandwidth^2))`. Bandwidth defaults to 1.0. Large vectors are subsampled to at most 1024 observations using a fixed seed. Aligned reference/prediction vectors use the same sampled cell indices, so identical predictions retain zero distance after sampling. The report stores bandwidth and sampling limit. This is a named, fully specified release metric, not an assertion that it equals every legacy script's unspecified distribution metric.

Do not compare flattened, cell-wise, and mean gene-wise correlations as if they were the same quantity. The package deliberately does not invent a definition for legacy manuscript labels such as “matrix distance” where the corresponding final analysis script has not been authenticated.
