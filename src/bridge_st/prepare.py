"""Import an existing HD research bundle without local server paths."""

import ast
import pickle
from pathlib import Path

import numpy as np

from .data import Bundle, save_bundle


def prepare_legacy(root, output, gene_file=None, trust_pickle=False, max_spots=None, gene_order="legacy"):
    """Convert the documented original on-disk format to portable NPZ arrays."""
    if not trust_pickle:
        raise ValueError("Legacy features.pkl is a pickle; explicitly pass --trust-features-pickle for your own trusted file.")
    import pandas as pd
    import scanpy as sc
    root = Path(root)
    required = [root / "binned_outputs/square_008um/filtered_feature_bc_matrix.h5", root / "binned_outputs/square_008um/spatial/tissue_positions.parquet", root / "preprocess/features.pkl", root / "preprocess/test_supernode_0.2.csv"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(str(path))
    if gene_file is None:
        gene_file = root / "top50_highly_expressed_genes_filtered.txt"
    genes = [x.strip() for x in Path(gene_file).read_text().splitlines() if x.strip()]
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("The gene panel must contain unique names in model-training order.")
    with required[2].open("rb") as handle:
        cached = pickle.load(handle)
    if not {"vit_feature", "pos_feature", "super_to_barcode"}.issubset(cached):
        raise ValueError("features.pkl must contain vit_feature, pos_feature, and super_to_barcode.")
    mapping = cached["super_to_barcode"]
    positions = pd.read_parquet(required[1]).set_index("barcode")
    adata = sc.read_10x_h5(required[0])
    adata.var_names_make_unique()
    missing = set(genes) - set(adata.var_names)
    if missing:
        raise ValueError(f"Panel genes absent from the matrix: {sorted(missing)}")
    if gene_order == "legacy":
        # Original utils.load_data uses a membership mask, retaining H5 order.
        genes = [str(x) for x in adata.var_names if x in set(genes)]
    elif gene_order != "panel":
        raise ValueError("gene_order must be legacy or panel")
    adata = adata[:, genes].copy()
    # Preserve caller-supplied training order; do not silently drop zero genes.
    cell_lookup = {str(name): i for i, name in enumerate(adata.obs_names)}
    feature_lookup = cached["vit_feature"]
    selected = sorted(str(x) for x in mapping)
    if max_spots is not None:
        if max_spots <= 0:
            raise ValueError("max_spots must be positive")
        selected = selected[:max_spots]
    spot_ids, all_indices, features, coordinates, cell_ids, spot_index = [], [], [], [], [], []
    counts = adata.X.tocsr()
    coarse_counts = []
    for spot in selected:
        nodes = mapping[spot]
        if isinstance(nodes, str):
            nodes = ast.literal_eval(nodes)
        valid_expression = [cell_lookup[str(name)] for name in nodes if str(name) in cell_lookup]
        valid = [str(name) for name in nodes if str(name) in cell_lookup and str(name) in feature_lookup and str(name) in positions.index]
        if not valid or not valid_expression:
            continue
        index = len(spot_ids)
        spot_ids.append(spot)
        # Coarse input includes every observed expression unit, including background.
        coarse_counts.append(np.asarray(counts[valid_expression].sum(axis=0)).ravel())
        for name in valid:
            cell_ids.append(name)
            all_indices.append(cell_lookup[name])
            spot_index.append(index)
            features.append(np.asarray(feature_lookup[name], dtype=np.float32).ravel())
            coordinates.append(positions.loc[name, ["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy(dtype=np.float32))
    if not spot_ids:
        raise ValueError("No matched expression/feature/position units were found.")
    def normalized(values):
        values = np.asarray(values, dtype=np.float32)
        sums = values.sum(axis=1, keepdims=True)
        return np.log1p(np.divide(values * 1e4, sums, out=np.zeros_like(values), where=sums != 0))
    truth = normalized(counts[all_indices].toarray())
    coarse = normalized(np.stack(coarse_counts))
    test_ids = set(required[3].read_text().splitlines())
    grid = np.array([[int(part) for part in name.split("_")] for name in spot_ids], dtype=np.float32)
    position_features = cached["pos_feature"].loc[cell_ids, ["x", "y"]].to_numpy(dtype=np.float32)
    bundle = Bundle(genes=np.array(genes), cell_ids=np.array(cell_ids), spot_ids=np.array(spot_ids), cell_spot_index=np.array(spot_index, dtype=np.int64), coordinates=np.array(coordinates, dtype=np.float32), image_features=np.stack(features), spot_expression=coarse, spot_coordinates=grid, test_spot_mask=np.array([name in test_ids for name in spot_ids]), cell_expression=truth, position_features=position_features)
    if Path(output).exists():
        raise ValueError(f"Output already exists: {output}")
    save_bundle(output, bundle)
    return Path(output)
