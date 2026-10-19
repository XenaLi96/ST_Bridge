# Preparing research inputs

## Upstream segmentation and features

Start from observed coarse expression, nucleus positions, and fixed per-nucleus image embeddings. Cell detection and image embedding extraction remain upstream processes. The original method refers to [CellViT](https://github.com/TIO-IKIM/CellViT); obtain that implementation and weights separately, record the exact version and input crop/coordinate transforms, and retain its upstream attribution and terms.

This package does not substitute randomly projected ResNet features for a CellViT-trained checkpoint. It does not automatically segment a whole-slide image or download a pretrained extractor. New raw-image pipelines must preserve the extractor, crop size, normalization, and coordinate convention expected by the chosen BRIDGE checkpoint.

## Existing server-layout HD inputs

`prepare-legacy` supports the following relative layout:

```text
sample/
  binned_outputs/square_008um/
    filtered_feature_bc_matrix.h5
    spatial/tissue_positions.parquet
  preprocess/
    features.pkl
    test_supernode_0.2.csv
  top50_highly_expressed_genes_filtered.txt
```

`features.pkl` is the original trusted cache with `vit_feature` (barcode to image vector), `pos_feature` (pandas DataFrame with normalized `x,y`), and `super_to_barcode` (coarse spot to all fine-bin barcodes). Every coarse-spot ID has the form `integer_integer`. The importer uses observed counts to rebuild the coarse sums and labels, rather than copying arbitrary cached normalization. It retains the supplied held-out split.

```bash
bridge-st prepare-legacy --root data/sample --out data/sample.npz \
  --trust-features-pickle
```

The pickle flag is required because this optional legacy format can contain executable Python objects. Use it only on a file whose provenance you trust; normal NPZ training/prediction does not use pickle input data.

The default `--gene-order legacy` retains H5 column order, matching the original loader's membership-mask selection. `--gene-order panel` instead retains the supplied panel order; use that when an already-confirmed checkpoint panel must be imposed on a second sample. Missing panel genes are an error, not silently removed.

```bash
bridge-st prepare-legacy --root data/second_sample --genes data/ordered_training_genes.txt \
  --gene-order panel --out data/second_sample.npz --trust-features-pickle
```

Prepare different samples into separate files. Do not infer tissue identity from historical one-letter directory aliases. Public 10x HD input measurements are available from the [10x Genomics datasets portal](https://www.10xgenomics.com/datasets); the exact manuscript accessions and image/feature provenance should accompany the final author-approved release.

For conventional spot data, construct the documented NPZ arrays directly from observed spots and detected nuclei. Set `cell_expression` absent and predict with a compatible trained checkpoint. The HD legacy importer is not a raw conventional-Visium preprocessing pipeline.
