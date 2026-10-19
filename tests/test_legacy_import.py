import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bridge_st.data import load_bundle


@unittest.skipUnless(importlib.util.find_spec("scanpy") is not None, "install the hd extra to test legacy conversion")
class LegacyImportTests(unittest.TestCase):
    def test_prepared_hd_format_and_gene_order(self):
        import h5py
        import pandas as pd
        from scipy.sparse import csc_matrix
        from bridge_st.prepare import prepare_legacy
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            matrix_dir = root / "binned_outputs/square_008um"
            (matrix_dir / "spatial").mkdir(parents=True)
            (root / "preprocess").mkdir()
            ids = [f"barcode_{i}" for i in range(12)]
            counts = np.array([[i + 1, 20 - i] for i in range(12)], dtype=np.float32)
            matrix = csc_matrix(counts.T)
            with h5py.File(matrix_dir / "filtered_feature_bc_matrix.h5", "w") as f:
                group = f.create_group("matrix")
                for key, value in {"data": matrix.data, "indices": matrix.indices, "indptr": matrix.indptr, "shape": matrix.shape, "barcodes": np.array(ids, dtype="S")}.items():
                    group.create_dataset(key, data=value)
                features = group.create_group("features")
                for key, value in {"name": ["GeneA", "GeneB"], "id": ["idA", "idB"], "feature_type": ["Gene Expression"] * 2, "genome": ["synthetic"] * 2}.items():
                    features.create_dataset(key, data=np.array(value, dtype="S"))
            positions = pd.DataFrame({"barcode": ids, "pxl_row_in_fullres": np.arange(12) * 5., "pxl_col_in_fullres": np.arange(12) * 6.})
            positions.to_parquet(matrix_dir / "spatial/tissue_positions.parquet")
            spot_names = [f"{i:05}_00000" for i in range(4)]
            mapping = {spot: ids[3 * i:3 * i + 3] for i, spot in enumerate(spot_names)}
            valid = [name for i, name in enumerate(ids) if i % 3 != 2]
            pos = pd.DataFrame([[.2, .8]] * len(valid), index=valid, columns=["x", "y"])
            cache = {"vit_feature": {name: np.ones(8, dtype=np.float32) for name in valid}, "pos_feature": pos, "super_to_barcode": mapping}
            with (root / "preprocess/features.pkl").open("wb") as f:
                pickle.dump(cache, f)
            (root / "preprocess/test_supernode_0.2.csv").write_text(spot_names[-1] + "\n")
            (root / "top50_highly_expressed_genes_filtered.txt").write_text("GeneB\nGeneA\n")
            with self.assertRaises(ValueError):
                prepare_legacy(root, root / "not_trusted.npz")
            prepare_legacy(root, root / "legacy.npz", trust_pickle=True)
            data = load_bundle(root / "legacy.npz", targets=True)
            self.assertEqual(data.genes.tolist(), ["GeneA", "GeneB"])
            self.assertEqual(len(data.cell_ids), 8)
            np.testing.assert_array_equal(data.position_features, pos.to_numpy(dtype=np.float32))
            expected = np.log1p(counts[:3].sum(0) / counts[:3].sum() * 1e4)
            np.testing.assert_allclose(data.spot_expression[0], expected, atol=1e-6)
            prepare_legacy(root, root / "panel.npz", trust_pickle=True, gene_order="panel")
            self.assertEqual(load_bundle(root / "panel.npz").genes.tolist(), ["GeneB", "GeneA"])


if __name__ == "__main__":
    unittest.main()
