"""Portable arrays for observed spots and nucleus-associated prediction units."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Bundle:
    genes: np.ndarray
    cell_ids: np.ndarray
    spot_ids: np.ndarray
    cell_spot_index: np.ndarray
    coordinates: np.ndarray
    image_features: np.ndarray
    spot_expression: np.ndarray
    spot_coordinates: np.ndarray
    test_spot_mask: np.ndarray
    cell_expression: object = None
    position_features: object = None

    def __post_init__(self):
        n, s, g = len(self.cell_ids), len(self.spot_ids), len(self.genes)
        if not n or not s or not g:
            raise ValueError("Cells, spots, and genes must all be nonempty.")
        for name in ("genes", "cell_ids", "spot_ids"):
            values = getattr(self, name)
            if values.ndim != 1 or values.dtype.kind not in "US" or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique one-dimensional string identifiers.")
        if self.cell_spot_index.shape != (n,) or self.cell_spot_index.dtype.kind not in "iu":
            raise ValueError("cell_spot_index must be an integer vector with one entry per cell.")
        if self.cell_spot_index.min() < 0 or self.cell_spot_index.max() >= s:
            raise ValueError("cell_spot_index contains an out-of-range spot index.")
        for name, shape in (("coordinates", (n, 2)), ("spot_coordinates", (s, 2)), ("spot_expression", (s, g))):
            values = getattr(self, name)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite with shape {shape}.")
        if self.image_features.ndim != 2 or self.image_features.shape[0] != n or self.image_features.shape[1] < 1 or not np.isfinite(self.image_features).all():
            raise ValueError("image_features must be a finite cells-by-features matrix.")
        if self.test_spot_mask.shape != (s,) or self.test_spot_mask.dtype.kind != "b":
            raise ValueError("test_spot_mask must be a boolean vector with one entry per spot.")
        if (self.spot_expression < 0).any():
            raise ValueError("spot_expression must be nonnegative.")
        if self.cell_expression is not None:
            if self.cell_expression.shape != (n, g) or not np.isfinite(self.cell_expression).all() or (self.cell_expression < 0).any():
                raise ValueError("cell_expression must be finite and nonnegative with shape (cells, genes).")
        if self.position_features is not None and (self.position_features.shape != (n, 2) or not np.isfinite(self.position_features).all()):
            raise ValueError("position_features must be finite with shape (cells, 2).")
        order = np.argsort(self.cell_spot_index, kind="stable")
        self.groups = [x for x in np.split(order, np.cumsum(np.bincount(self.cell_spot_index, minlength=s))[:-1])]
        self._edge_cache = {}

    @property
    def node_features(self):
        low = self.coordinates.min(axis=0)
        span = np.maximum(self.coordinates.max(axis=0) - low, 1e-12)
        position = (self.coordinates - low) / span if self.position_features is None else self.position_features
        return np.concatenate((self.image_features, position), axis=1).astype(np.float32)

    def local_edges(self, spot, radius=150.0):
        key = (int(spot), float(radius))
        if key not in self._edge_cache:
            points = self.coordinates[self.groups[spot]]
            pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
            if pairs.size:
                # The research dataset uses a strict distance < radius.
                pairs = pairs[np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1) < radius]
            edges = np.concatenate((pairs, pairs[:, ::-1]), axis=0).T
            self._edge_cache[key] = np.asarray(edges, dtype=np.int64).reshape(2, -1)
        return self._edge_cache[key]

    @property
    def spot_edges(self):
        lookup = {tuple(x): i for i, x in enumerate(self.spot_coordinates)}
        if len(lookup) != len(self.spot_ids):
            raise ValueError("spot_coordinates must be unique.")
        edges = []
        for i, (x, y) in enumerate(self.spot_coordinates):
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                j = lookup.get((x + dx, y + dy))
                if j is not None:
                    edges.append((i, j))
        return np.asarray(edges, dtype=np.int64).reshape(-1, 2).T


def load_bundle(path, targets=False):
    """Never access the cell-expression array when loading prediction inputs."""
    names = ("genes", "cell_ids", "spot_ids", "cell_spot_index", "coordinates", "image_features", "spot_expression", "spot_coordinates", "test_spot_mask")
    with np.load(path, allow_pickle=False) as z:
        missing = set(names) - set(z.files)
        if missing:
            raise ValueError(f"Missing input arrays: {', '.join(sorted(missing))}")
        values = {name: z[name] for name in names}
        if "position_features" in z:
            values["position_features"] = z["position_features"]
        if targets:
            if "cell_expression" not in z:
                raise ValueError("Training/evaluation requires cell_expression labels.")
            values["cell_expression"] = z["cell_expression"]
    return Bundle(**values)


def save_bundle(path, bundle, targets=True):
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Data bundles must use the .npz extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = ("genes", "cell_ids", "spot_ids", "cell_spot_index", "coordinates", "image_features", "spot_expression", "spot_coordinates", "test_spot_mask")
    values = {name: getattr(bundle, name) for name in names}
    if bundle.position_features is not None:
        values["position_features"] = bundle.position_features
    if targets and bundle.cell_expression is not None:
        values["cell_expression"] = bundle.cell_expression
    np.savez_compressed(path, **values)


def make_demo(directory, seed=7):
    """Generate explicitly synthetic, small, non-patient example inputs."""
    rng = np.random.default_rng(seed)
    s, n_per, g, f = 16, 5, 6, 384
    spot_coordinates = np.array([(x, y) for x in range(4) for y in range(4)], dtype=np.float32)
    spot_index = np.repeat(np.arange(s), n_per)
    image_features = rng.normal(size=(s * n_per, f)).astype(np.float32)
    coordinates = spot_coordinates[spot_index] * 80 + rng.uniform(0, 40, size=(s * n_per, 2))
    expression = np.maximum(0, image_features[:, :g] * 1.2 + rng.normal(1, .2, (s * n_per, g))).astype(np.float32)
    coarse = np.stack([expression[spot_index == i].mean(axis=0) for i in range(s)])
    mask = np.zeros(s, dtype=bool)
    mask[rng.choice(s, 4, replace=False)] = True
    bundle = Bundle(genes=np.array([f"SyntheticGene{i}" for i in range(g)]), cell_ids=np.array([f"synthetic_cell_{i}" for i in range(s * n_per)]), spot_ids=np.array([f"{x:05}_{y:05}" for x, y in spot_coordinates.astype(int)]), cell_spot_index=spot_index, coordinates=coordinates.astype(np.float32), image_features=image_features, spot_expression=coarse, spot_coordinates=spot_coordinates, test_spot_mask=mask, cell_expression=expression)
    directory = Path(directory)
    save_bundle(directory / "synthetic.npz", bundle)
    save_bundle(directory / "synthetic_inputs.npz", bundle, targets=False)
    return directory
