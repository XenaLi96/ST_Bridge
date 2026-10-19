# Data format

An input is a compressed NumPy `.npz` archive with the following arrays. All identifiers are unique Unicode/string arrays; object arrays and executable pickle payloads are not accepted by the core loader.

| Array | Shape | Meaning |
| --- | --- | --- |
| `genes` | G | Exact ordered gene names. |
| `cell_ids` | N | Unique nucleus-associated prediction-unit identifiers. |
| `spot_ids` | S | Unique coarse-spot identifiers. |
| `cell_spot_index` | N | Integer spot index for each cell, in `[0,S)`. |
| `coordinates` | N × 2 | Finite spatial coordinates used for within-spot radius edges and output locations. |
| `image_features` | N × F | Fixed image embeddings; research checkpoints commonly use F=384. |
| `spot_expression` | S × G | Nonnegative observed coarse expression, on the scale used for training. |
| `spot_coordinates` | S × 2 | Coarse grid coordinates; hierarchical edges connect offsets `(±1,0),(0,±1)`. |
| `test_spot_mask` | S | Boolean held-out membership. Required even for prediction; use all false for an unlabeled deployment with `--split all`. |
| `cell_expression` | N × G | Nonnegative reference labels, required only by train/evaluate. |
| `position_features` | N × 2 | Optional positional embeddings. If absent, coordinates are independently min-max scaled per input sample. |

The normal-node encoder receives `[image_features, position_features]`. The original HD importer retains the cached normalized positional features to preserve checkpoint compatibility; raw spatial coordinates are still used for the local edge radius. A radius of 150 only has the original meaning when coordinates use the same pixel convention. Do not supply micrometers to a pixel-trained model without an explicit conversion.

Cell-expression labels are loaded only when `targets=True`. Prediction reads no target array, including when a labeled bundle is supplied. Training losses and visual-cluster expression prototypes use training spots only. The hierarchical graph may see observed coarse expression at held-out spots (transductive observed inputs), but not their cell-expression targets.

The original HD normalization is `log1p(10000 * counts / row_sum)` after selecting the gene panel. Coarse counts are summed over all mapped expression units, including units without nucleus image features; only matched nucleus-associated units become prediction rows. Replacing this scale with raw counts or normalizing over a different panel changes model inputs.

Prediction archives contain `predictions[N,G]`, `cell_ids`, `genes`, `coordinates`, and `spot_ids` for the selected rows. They contain no target labels. Evaluation reorders rows by cell ID, requires all rows in the requested split, and rejects unmatched gene order.
