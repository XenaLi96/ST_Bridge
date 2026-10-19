import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bridge_st.data import load_bundle, make_demo, save_bundle
from bridge_st.engine import convert_checkpoint, load_model, predict, train
from bridge_st.metrics import evaluate, mmd_rbf_squared, regression_metrics
from bridge_st.network import Config


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_demo(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, **values):
        base = {"epochs": 1, "hidden_dim": 8, "batch_size": 4, "seed": 19}
        base.update(values)
        return Config.from_dict(base)

    def test_label_free_prediction_and_alignment(self):
        cfg = self.config(expr_head=True, expr_gate=True, expr_gate_hard=True, celltype=True, celltype_k=3, quantile_lambda=1, rate_lambda=1)
        ckpt = train(self.root / "synthetic.npz", self.root / "run", cfg)
        a = predict(self.root / "synthetic.npz", ckpt, self.root / "a.npz")
        b = predict(self.root / "synthetic_inputs.npz", ckpt, self.root / "b.npz")
        np.testing.assert_array_equal(np.load(a)["predictions"], np.load(b)["predictions"])
        with np.load(a, allow_pickle=False) as z:
            arrays = {name: z[name][::-1] if name != "genes" else z[name] for name in z.files}
        np.savez_compressed(self.root / "reordered.npz", **arrays)
        first = evaluate(self.root / "synthetic.npz", a, self.root / "metrics1", max_samples=32)
        second = evaluate(self.root / "synthetic.npz", self.root / "reordered.npz", self.root / "metrics2", max_samples=32)
        self.assertEqual(json.loads(first.read_text()), json.loads(second.read_text()))
        self.assertEqual(json.loads(first.read_text())["n_cells"], 20)

    def test_held_out_labels_do_not_affect_training(self):
        data = load_bundle(self.root / "synthetic.npz", targets=True)
        data.cell_expression[data.test_spot_mask[data.cell_spot_index]] = 9999
        save_bundle(self.root / "perturbed.npz", data)
        cfg = self.config(celltype=True, celltype_k=3, expr_head=True, expr_gate=True, rate_lambda=1, quantile_lambda=1, variance_lambda=.2, rank_lambda=.2)
        one = train(self.root / "synthetic.npz", self.root / "one", cfg)
        two = train(self.root / "perturbed.npz", self.root / "two", cfg)
        a = torch.load(one, weights_only=True)["state_dict"]
        b = torch.load(two, weights_only=True)["state_dict"]
        for name in a:
            self.assertTrue(torch.equal(a[name], b[name]), name)

    def test_hierarchical_mode_and_checkpoint_conversion(self):
        cfg = self.config(mode="hierarchical")
        ckpt = train(self.root / "synthetic.npz", self.root / "h", cfg)
        model, metadata = load_model(ckpt, torch.device("cpu"))
        legacy = {"model_state_dict": model.graph.state_dict(), "noise_predictor_state_dict": model.background_predictor.state_dict(), "normal_encoder_state_dict": model.normal_encoder.state_dict()}
        torch.save(legacy, self.root / "legacy.pth")
        (self.root / "genes.txt").write_text("\n".join(metadata["genes"]))
        converted = convert_checkpoint(self.root / "legacy.pth", self.root / "genes.txt", cfg, self.root / "converted.pt")
        a = predict(self.root / "synthetic_inputs.npz", ckpt, self.root / "hpred.npz")
        b = predict(self.root / "synthetic_inputs.npz", converted, self.root / "hconv.npz")
        np.testing.assert_array_equal(np.load(a)["predictions"], np.load(b)["predictions"])
        self.assertTrue(np.isfinite(np.load(a)["predictions"]).all())

    def test_c2f_checkpoint_conversion_and_optional_heads(self):
        cfg = self.config(expr_head=True, expr_gate=True, expr_gate_hard=True, use_node_gnn=True, use_pos_head=True)
        ckpt = train(self.root / "synthetic.npz", self.root / "c", cfg)
        model, metadata = load_model(ckpt, torch.device("cpu"))
        legacy = {"normal_encoder_state_dict": model.normal_encoder.state_dict(), "refiner_state_dict": model.refiner.state_dict(), "expr_head_state_dict": {"head." + k: v for k, v in model.expr_head.state_dict().items()}, "node_gnn_state_dict": model.node_gnn.state_dict(), "pos_head_state_dict": model.pos_head.state_dict(), "expr_gate": True, "expr_gate_hard": True, "expr_gate_thr": .5}
        torch.save(legacy, self.root / "legacy.pth")
        (self.root / "genes.txt").write_text("\n".join(metadata["genes"]))
        converted = convert_checkpoint(self.root / "legacy.pth", self.root / "genes.txt", cfg, self.root / "converted.pt")
        a = predict(self.root / "synthetic_inputs.npz", ckpt, self.root / "cpred.npz")
        b = predict(self.root / "synthetic_inputs.npz", converted, self.root / "cconv.npz")
        np.testing.assert_array_equal(np.load(a)["predictions"], np.load(b)["predictions"])

    def test_bad_gene_order_and_overwrite_are_rejected(self):
        cfg = self.config()
        ckpt = train(self.root / "synthetic.npz", self.root / "run", cfg)
        with self.assertRaises(ValueError):
            train(self.root / "synthetic.npz", self.root / "run", cfg)
        data = load_bundle(self.root / "synthetic_inputs.npz")
        data.genes = data.genes[::-1]
        save_bundle(self.root / "bad.npz", data, targets=False)
        with self.assertRaisesRegex(ValueError, "gene names/order"):
            predict(self.root / "bad.npz", ckpt, self.root / "badpred.npz")

    def test_single_cell_empty_edges_and_explicit_positions(self):
        data = load_bundle(self.root / "synthetic.npz", targets=True)
        one = copy.deepcopy(data)
        for name in ("cell_ids", "coordinates", "image_features", "cell_expression"):
            setattr(one, name, getattr(one, name)[:1])
        one.cell_spot_index = np.zeros(1, dtype=np.int64)
        one.position_features = np.array([[.3, .7]], dtype=np.float32)
        one.__post_init__()
        self.assertEqual(one.local_edges(0).shape, (2, 0))
        np.testing.assert_array_equal(one.node_features[0, -2:], one.position_features[0])

    def test_metric_identity_and_constant_vectors(self):
        x = np.array([0., 1., 2., 5.])
        metrics = regression_metrics(x, x)
        self.assertEqual(metrics["mae"], 0.)
        self.assertAlmostEqual(metrics["pearson"], 1.)
        self.assertAlmostEqual(metrics["r2"], 1.)
        self.assertAlmostEqual(mmd_rbf_squared(x, x), 0.)
        larger = np.linspace(0, 5, 100)
        self.assertEqual(mmd_rbf_squared(larger, larger, max_samples=16), 0.)
        self.assertIsNone(regression_metrics(np.ones(3), np.ones(3))["r2"])

    def test_invalid_configuration(self):
        for values in ({"expr_gate": True}, {"epochs": 0}, {"edge_radius": 0}, {"unknown": 3}, {"mode": "other"}):
            with self.assertRaises(ValueError):
                Config.from_dict(values)


if __name__ == "__main__":
    unittest.main()
