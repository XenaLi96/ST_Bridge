"""Explicit regression, activation, and distribution metrics; no GT calibration."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import energy_distance, spearmanr

from .data import load_bundle


def regression_metrics(truth, pred):
    x, y = np.asarray(truth, dtype=float).ravel(), np.asarray(pred, dtype=float).ravel()
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Metrics require nonempty finite reference and prediction arrays.")
    if x.shape != y.shape:
        raise ValueError("Prediction and reference shapes differ.")
    error = y - x
    mse = float(np.mean(error ** 2))
    variable_x = len(x) > 1 and np.ptp(x) > 0
    variable_y = len(y) > 1 and np.ptp(y) > 0
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return {"mae": float(np.mean(np.abs(error))), "mse": mse, "rmse": mse ** .5, "pearson": float(np.corrcoef(x, y)[0, 1]) if variable_x and variable_y else None, "spearman": float(spearmanr(x, y).statistic) if variable_x and variable_y else None, "r2": float(1 - np.sum(error ** 2) / np.sum((x - x.mean()) ** 2)) if variable_x else None, "cosine": float(np.dot(x, y) / denominator) if denominator > 0 else None}


def activation_metrics(truth, pred, threshold):
    a, b = truth > threshold, pred > threshold
    tp, fp, fn = np.sum(a & b), np.sum(~a & b), np.sum(a & ~b)
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    return {"accuracy": float(np.mean(a == b)), "precision": precision, "recall": recall, "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0, "reference_activation_rate": float(a.mean()), "predicted_activation_rate": float(b.mean()), "activation_rate_difference": float(abs(a.mean() - b.mean()))}


def mmd_rbf_squared(x, y, bandwidth=1.0, max_samples=1024, seed=0):
    if bandwidth <= 0 or max_samples < 2:
        raise ValueError("MMD bandwidth must be positive and max_samples at least 2.")
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    if len(x) == len(y) and len(x) > max_samples:
        # Reference and predictions describe aligned cells: sample the same cells.
        indices = rng.choice(len(x), max_samples, replace=False)
        x, y = x[indices], y[indices]
    elif len(x) > max_samples:
        x = x[rng.choice(len(x), max_samples, replace=False)]
    if len(y) > max_samples:
        y = y[rng.choice(len(y), max_samples, replace=False)]
    kernel = lambda a, b: np.exp(-((a[:, None] - b[None, :]) ** 2) / (2 * bandwidth ** 2)).mean()
    return float(max(0, kernel(x, x) + kernel(y, y) - 2 * kernel(x, y)))


def aligned_arrays(reference_path, predictions_path, split):
    bundle = load_bundle(reference_path, targets=True)
    with np.load(predictions_path, allow_pickle=False) as z:
        genes, ids, predictions = z["genes"], z["cell_ids"], z["predictions"]
    if genes.tolist() != bundle.genes.tolist():
        raise ValueError("Prediction gene order does not match the reference panel.")
    if len(set(ids)) != len(ids) or predictions.shape != (len(ids), len(genes)):
        raise ValueError("Predictions need unique cell IDs and a cells-by-genes array.")
    wanted = np.ones(len(bundle.cell_ids), dtype=bool) if split == "all" else bundle.test_spot_mask[bundle.cell_spot_index]
    if split not in {"all", "test"} or not wanted.any():
        raise ValueError("Evaluation split must be a nonempty all or test split.")
    lookup = {name: i for i, name in enumerate(ids)}
    expected = bundle.cell_ids[wanted]
    if any(name not in lookup for name in expected):
        raise ValueError("Predictions do not cover every cell in the requested evaluation split.")
    if not set(ids).issubset(set(bundle.cell_ids)):
        raise ValueError("Predictions contain cell IDs absent from the reference.")
    return bundle.genes, bundle.cell_expression[wanted], predictions[[lookup[name] for name in expected]]


def evaluate(reference, predictions, output_dir, split="test", threshold=.3, bandwidth=1.0, max_samples=1024):
    genes, truth, pred = aligned_arrays(reference, predictions, split)
    if not np.isfinite(pred).all():
        raise ValueError("Predictions contain non-finite values.")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_gene = []
    for i, gene in enumerate(genes):
        row = {"gene": str(gene), **regression_metrics(truth[:, i], pred[:, i]), **activation_metrics(truth[:, i], pred[:, i], threshold)}
        row["energy_distance"] = float(energy_distance(truth[:, i], pred[:, i]))
        row["mmd_rbf_squared"] = mmd_rbf_squared(truth[:, i], pred[:, i], bandwidth, max_samples)
        per_gene.append(row)
    with (output_dir / "per_gene_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_gene[0]))
        writer.writeheader()
        writer.writerows(per_gene)
    means = {}
    for name in per_gene[0]:
        if name == "gene":
            continue
        values = [x[name] for x in per_gene if x[name] is not None and np.isfinite(x[name])]
        means[name] = float(np.mean(values)) if values else None
    summary = {"split": split, "n_cells": len(truth), "n_genes": len(genes), "activation_threshold": threshold, "mmd_bandwidth": bandwidth, "mmd_max_samples": max_samples, "flattened_regression": regression_metrics(truth, pred), "pooled_activation": activation_metrics(truth, pred, threshold), "mean_gene_metrics": means, "undefined_correlations": "null in JSON; empty in CSV; omitted from finite gene means"}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return output_dir / "summary.json"
