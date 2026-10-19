# Recovered sources and release refactor

The source manifest records SHA256 hashes of the original research files using project-relative paths. `models.py` and `refinement.py` are byte-identical copies of the recovered model-definition files. The two original training scripts are preserved as reference text under `recovered_sources/`; they are not runnable entrypoints in this directory.

The older `HGNN4ST/bridge` folder was a 2025 snapshot with a simpler C2F trainer. The primary C2F recipe here was recovered from the later trainer and H/shared-panel cell-type/quantile job configurations. This selection remains subject to final author confirmation; these files do not establish provenance for every historical figure/table.

## Intentional engineering and correctness changes

1. Inputs, outputs, configuration, and device are CLI arguments. No personal filesystem paths, cluster node names, credentials, notification hooks, or Slurm requirements are part of the runnable package.
2. The portable format is numeric/string NPZ. The optional HD importer preserves original H5 gene order by default and cached positional features, reconstructs coarse sums, and requires an explicit trusted-pickle flag for the existing feature cache.
3. The recovered C2F trainer iterates sampler `n_id` without excluding held-out spot labels. The release excludes those labels and trains visual-cluster expression prototypes only on training cells. Its training split is therefore an intentional correction, **not identical to that historical script**.
4. Spots are visited once per epoch. The release avoids repeated neighbor-expanded visits, dimension constants of 50/386, and `.squeeze()` failures for a single node. Radius edges and four-neighbor coarse-grid edges retain their original definitions. GCN self-loops permit isolated prediction units.
5. The hierarchical implementation retains the original model layers, background construction, readout alignment, and reconstruction terms but uses full observed coarse-graph passes with configurable batches of training spots, rather than the old `NeighborSampler` schedule. The example hierarchical configuration is not a recovered final run specification.
6. Model train/eval mode is explicit. Prediction uses checkpoint heads/gate settings and never cell-expression labels. GT-derived threshold optimization, GT mean alignment, and plotting-time intensity corrections from legacy visualization scripts are not included in prediction.
7. The C2F global `GraphAutoencoder` was instantiated and saved but unused in the recovered C2F forward pass. The release's C2F model omits that unused module; the distinct hierarchical mode includes it. No extra global-graph learning is claimed for C2F.
8. The original C2F losses are retained: Smooth-L1, mean/sum consistency with annealing, optional L1, expression BCE and activation-rate penalty, variance/quantile terms, visual-cluster prototype MSE, pairwise ranking, and marker MSE/KL/CE. For a one-cell group, variance matching is skipped to avoid an undefined sample standard deviation.

These changes support portable execution and explicit hold-out evaluation. They require rerunning real experiments before attaching this refactor's results to historical manuscript numbers. Synthetic checks validate software behavior and checkpoint conversion, not biological performance. No real trained weights or final table-output provenance are asserted by this archive.
