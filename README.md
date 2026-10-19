# BRIDGE

Portable research code for nucleus-associated spatial gene-expression prediction.

This release candidate contains the recovered hierarchical graph model and the later coarse-to-fine (C2F) refinement workflow, packaged with explicit inputs, training splits, prediction, evaluation, and checkpoint conversion. The original model-definition files are preserved byte for byte; source hashes and the recovered training scripts are included under `provenance/`.

**Release status:** `0.1.0rc1`, prepared on 2026-10-17. Synthetic end-to-end validation is documented in [docs/validation.md](docs/validation.md). Final authors' selection of the manuscript code snapshot and the software license remains to be confirmed. See [LICENSING.md](LICENSING.md). This is a portable refactor, **not a claim that every manuscript table has been reproduced**. No manuscript numerical results were changed.

## Installation

Tested with Python 3.9.19, PyTorch 2.4.1, PyG 2.5.3, NumPy 1.26.4, SciPy 1.13.1, and scikit-learn 1.5.2. Python 3.9–3.11 is accepted by the package; only Python 3.9 was exercised here.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
# Optional: importing existing HD research data also needs the hd dependencies.
python -m pip install '.[hd]'
```

For a selected CUDA build, install the appropriate PyTorch 2.4.1 wheel before this package, following the [official previous-version instructions](https://pytorch.org/get-started/previous-versions/). The runtime does not use `NeighborSampler`, `torch-sparse`, or `torch-scatter`; the supported GCN operations work with PyG's base installation. See the [PyG 2.5.3 installation documentation](https://pytorch-geometric.readthedocs.io/en/2.5.3/install/installation.html). No model or dataset is automatically downloaded.

## Five-minute synthetic example

Run from this source directory after installation. The generated data are entirely synthetic and carry no patient information.

```bash
bridge-st make-demo --out data/demo
bridge-st validate --data data/demo/synthetic.npz --targets
bridge-st train --data data/demo/synthetic.npz --config configs/demo.json \
  --out outputs/demo --device cpu
bridge-st predict --data data/demo/synthetic_inputs.npz \
  --checkpoint outputs/demo/model.pt --out outputs/demo_predictions.npz --device cpu
bridge-st evaluate --reference data/demo/synthetic.npz \
  --predictions outputs/demo_predictions.npz --out outputs/demo_metrics --split test
```

Training produces `model.pt`, `run.json` (configuration, software version, input hash, and split sizes), and `training.jsonl`. Evaluation produces `summary.json` and `per_gene_metrics.csv`. Output paths must be new; existing runs are not overwritten.

## Inputs and supervision

Training requires **cell-level reference expression**. Prediction requires observed spot expression, nucleus-associated image features, nucleus coordinates, spot membership, and the trained gene panel. **Prediction does not require or read cell-expression labels.** It also works with a label-free input file, as shown above.

The required NPZ arrays and their units are specified in [docs/data-format.md](docs/data-format.md). NPZ files use numeric/string arrays and are loaded with `allow_pickle=False`. Gene names and their order must match the checkpoint exactly.

The segmentation and feature extractor are upstream inputs. This source package does not embed CellViT, its weights, tissue images, or expression datasets. Use a fixed, documented extractor compatible with the checkpoint; changing the extractor or feature scale is a model-input change. [Data preparation](docs/data-preparation.md) describes the original HD layout and a converter.

## Models and training configurations

| Mode | Configuration | Implementation |
| --- | --- | --- |
| C2F residual baseline | `configs/c2f_baseline.json` | Morphology/position encoder plus a residual refinement of observed spot expression. |
| C2F cell-type prior and quantile matching | `configs/c2f_celltype_quantile.json` | The later research recipe: visual-cluster expression prototypes, expression gate, activation-rate penalty, and quantile loss. |
| Hierarchical graph model | `configs/hierarchical.json` | Spot graph autoencoder, shared cell decoder, background predictor, and reconstruction/readout/consistency losses. |

The hierarchical and C2F entrypoints are **distinct recovered implementations**. The C2F research trainer instantiates a global autoencoder but does not use it in its forward pass; this package does not silently describe C2F as training that global graph. The provided hierarchical configuration uses manuscript-style learning rate/batch values as an example, not an authenticated historical experiment configuration. See [provenance/release-changes.md](provenance/release-changes.md).

The C2F `celltype` option uses visual-cluster expression prototypes estimated from **training cells**, not externally annotated ground-truth cell-type labels. `marker_sets` can separately enable the marker-score MSE/KL/cross-entropy objectives from the recovered trainer. Marker sets are explicit gene-name lists supplied in the JSON configuration; at least two sets must overlap the training panel.

```bash
bridge-st train --data data/sample.npz --config configs/c2f_celltype_quantile.json \
  --out outputs/sample --device cuda
bridge-st train --data data/sample.npz --config configs/hierarchical.json \
  --out outputs/hierarchical --device cuda
```

All configuration fields are defined by `bridge_st.network.Config`. Unknown keys are rejected. The spatial radius is expressed in the input coordinate units; the recovered default is 150 pixels. C2F `batch_size=1` retains one optimizer step per spot. A larger value averages losses over that many spots before an optimizer step.

## Existing data and trained checkpoints

```bash
bridge-st prepare-legacy --root data/original_hd_sample \
  --out data/sample.npz --trust-features-pickle

# Export the ACTUAL ordered panel from prepared training data.
python -c 'import numpy as np; z=np.load("data/sample.npz"); open("data/ordered_genes.txt","w").write("\n".join(z["genes"].tolist())+"\n")'

bridge-st convert-checkpoint --source checkpoints/research_model.pth \
  --genes data/ordered_genes.txt --config configs/c2f_celltype_quantile.json \
  --out checkpoints/research_model_portable.pt
```

The original ranked top-gene text file may **not** be the model's column order: the legacy loader retains H5 gene order. Conversion requires the exact training order, correct mode/head settings, and weights from the corresponding source version. State dicts are loaded strictly. Do not guess a panel or mix checkpoints from incompatible feature extractors. No real pretrained weights are bundled or implied by the synthetic example.

## Evaluation and reproducibility

Evaluation defaults to held-out spots (`--split test`). `--split all` is explicit. Predictions are aligned to reference cells by ID, not row position. The report separates flattened regression, pooled activation, and mean gene-wise metrics. [Metric definitions](docs/metrics.md) describe thresholds, constant vectors, MMD sampling, and aggregation.

This port fixes several research-script problems: C2F cell labels from held-out spots are excluded from training, spots are visited once per epoch rather than repeatedly through sampled neighbors, inference uses eval mode, and GT-dependent threshold/mean calibration is omitted from prediction. Consequently, retrained portable models are **not guaranteed to reproduce historical checkpoints or manuscript numbers**. Historical figure provenance and original experiment partitions still require author review; see [release notes](RELEASE_NOTES.md).

## Tests and source archives

```bash
python -m unittest discover -s tests -v
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
python scripts/make_release.py
```

The archive builder includes source, configurations, docs, reference training text, and tests. It excludes checkpoints, datasets, caches, environment files, notification credentials, and experimental result folders. A `SHA256SUMS` file is included. Nothing is pushed to a remote repository by these commands.
