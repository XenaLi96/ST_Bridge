"""Training and label-independent prediction for the recovered research models."""

import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

from . import __version__
from .data import load_bundle
from .network import BridgeNetwork, Config


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; select --device cpu.")
    return torch.device(name)


def marker_indices(genes, marker_sets):
    if not marker_sets:
        return []
    lookup = {g: i for i, g in enumerate(genes)}
    indices = [[lookup[g] for g in values if g in lookup] for values in marker_sets.values()]
    indices = [x for x in indices if x]
    if len(indices) < 2:
        raise ValueError("Marker supervision requires at least two marker sets overlapping the gene panel.")
    return indices


def marker_scores(expression, indices):
    expression = torch.log1p(expression.clamp(min=0))
    return torch.stack([expression[:, group].mean(dim=1) for group in indices], dim=1)


def c2f_loss(pred, truth, coarse, logits, config, consistency, marker_idx, prior=None):
    loss = config.node_loss_weight * F.smooth_l1_loss(pred, truth)
    aggregate = pred.sum(0) if config.use_sum else pred.mean(0)
    target = coarse * pred.shape[0] if config.use_sum else coarse
    loss = loss + consistency * F.mse_loss(aggregate, target)
    if config.l1_lambda:
        loss = loss + config.l1_lambda * pred.abs().mean()
    if logits is not None:
        active = (truth > config.expression_threshold).to(truth.dtype)
        loss = loss + config.expr_lambda * F.binary_cross_entropy_with_logits(logits, active)
        if config.rate_lambda:
            loss = loss + config.rate_lambda * F.mse_loss(logits.sigmoid().mean(0), active.mean(0))
    if config.variance_lambda and pred.shape[0] > 1:
        loss = loss + config.variance_lambda * F.mse_loss(pred.std(0), truth.std(0))
    if config.quantile_lambda:
        terms = [F.mse_loss(torch.quantile(pred, q, dim=0), torch.quantile(truth, q, dim=0)) for q in config.quantiles]
        loss = loss + config.quantile_lambda * torch.stack(terms).mean()
    if marker_idx:
        a, b = marker_scores(pred, marker_idx), marker_scores(truth, marker_idx)
        loss = loss + config.marker_lambda * F.mse_loss(a, b)
        temperature = config.marker_temperature
        loss = loss + config.marker_kl_lambda * F.kl_div(F.log_softmax(a / temperature, 1), F.softmax(b / temperature, 1), reduction="batchmean") * temperature ** 2
        loss = loss + config.marker_ce_lambda * F.cross_entropy(a, b.argmax(1))
    if prior is not None:
        loss = loss + config.celltype_lambda * F.mse_loss(pred, prior)
    if config.rank_lambda and pred.shape[0] > 1:
        n = pred.shape[0]
        k = min(config.rank_pairs, n * (n - 1) // 2)
        i, j = torch.randint(n, (k,), device=pred.device), torch.randint(n, (k,), device=pred.device)
        valid = i != j
        delta = truth[i[valid]] - truth[j[valid]]
        difference = pred[i[valid]] - pred[j[valid]]
        keep = delta.abs() > config.rank_epsilon
        if keep.any():
            loss = loss + config.rank_lambda * F.softplus(-delta.sign() * difference / config.rank_tau)[keep].mean()
    return loss


def _celltype_prior(bundle, features, config, rng):
    # Only training cells can affect clustering and expression prototypes.
    indices = np.flatnonzero(~bundle.test_spot_mask[bundle.cell_spot_index])
    if config.celltype_sample > 0 and len(indices) > config.celltype_sample:
        indices = rng.choice(indices, config.celltype_sample, replace=False)
    if len(indices) < config.celltype_k:
        raise ValueError("Not enough training cells for celltype_k clusters.")
    kmeans = MiniBatchKMeans(n_clusters=config.celltype_k, random_state=config.seed, batch_size=4096, n_init="auto")
    labels = kmeans.fit_predict(features[indices])
    means = np.zeros((config.celltype_k, len(bundle.genes)), dtype=np.float32)
    for k in range(config.celltype_k):
        if (labels == k).any():
            means[k] = bundle.cell_expression[indices[labels == k]].mean(0)
    return kmeans, means


def train(data_path, output_dir, config, device="cpu"):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}; choose a new run directory.")
    bundle = load_bundle(data_path, targets=True)
    train_spots = np.array([i for i, group in enumerate(bundle.groups) if len(group) and not bundle.test_spot_mask[i]])
    test_spots = [i for i, group in enumerate(bundle.groups) if len(group) and bundle.test_spot_mask[i]]
    if not len(train_spots) or not test_spots:
        raise ValueError("Training requires nonempty training and held-out spot groups.")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.default_rng(config.seed)
    device = choose_device(device)
    features = bundle.node_features
    model = BridgeNetwork(config, len(bundle.genes), features.shape[1]).to(device)
    optimizer_class = torch.optim.Adam if config.mode == "hierarchical" else torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    marker_idx = marker_indices(bundle.genes.tolist(), config.marker_sets)
    prior = _celltype_prior(bundle, features, config, rng) if config.celltype else None
    coarse_all = torch.as_tensor(bundle.spot_expression, dtype=torch.float32, device=device)
    spot_edges = torch.as_tensor(bundle.spot_edges, dtype=torch.long, device=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    run = {"version": __version__, "config": config.to_dict(), "data_filename": Path(data_path).name, "data_sha256": file_sha256(data_path), "python": platform.python_version(), "torch": str(torch.__version__), "device": str(device), "training_spots": len(train_spots), "held_out_spots": len(test_spots), "training_cells": int(sum(len(bundle.groups[i]) for i in train_spots))}
    (output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    with (output_dir / "training.jsonl").open("w") as log:
        for epoch in range(config.epochs):
            model.train()
            order = rng.permutation(train_spots)
            consistency = float(np.interp(epoch, [0, max(config.epochs - 1, 1)], [1, .1])) * config.consistency_scale
            losses = []
            for start in range(0, len(order), config.batch_size):
                batch = order[start:start + config.batch_size]
                optimizer.zero_grad()
                if config.mode == "hierarchical":
                    z, reconstruction = model.graph(coarse_all, [spot_edges, spot_edges])
                    loss = F.mse_loss(reconstruction, coarse_all)
                else:
                    loss = None
                for spot in batch:
                    cells = bundle.groups[spot]
                    feature = torch.as_tensor(features[cells], device=device)
                    truth = torch.as_tensor(bundle.cell_expression[cells], dtype=torch.float32, device=device)
                    edges = torch.as_tensor(bundle.local_edges(spot, config.edge_radius), dtype=torch.long, device=device)
                    pred, hidden, logits = model.forward_group(feature, coarse_all[spot], edges)
                    if config.mode == "hierarchical":
                        background = F.softplus(model.background_predictor(z[spot]))
                        combined = torch.log1p(F.softplus(pred).sum(0) + background + 1e-8)
                        term = F.mse_loss(pred, truth) + F.mse_loss(hidden.mean(0), z[spot]) + F.mse_loss(combined, coarse_all[spot])
                    else:
                        target = None
                        if prior is not None:
                            clusters = prior[0].predict(features[cells])
                            target = torch.as_tensor(prior[1][clusters], device=device)
                        term = c2f_loss(pred, truth, coarse_all[spot], logits, config, consistency, marker_idx, target)
                    loss = term if loss is None else loss + term
                # C2F originally steps per spot; configure batch_size=1 for that schedule.
                if config.mode == "c2f":
                    loss = loss / len(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError("Training produced a non-finite loss; inspect input scale and configuration.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            record = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "consistency_weight": consistency}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
    checkpoint = {"format": "bridge-st-v1", "version": __version__, "config": config.to_dict(), "genes": bundle.genes.tolist(), "n_features": int(features.shape[1]), "state_dict": model.cpu().state_dict()}
    torch.save(checkpoint, output_dir / "model.pt")
    return output_dir / "model.pt"


def load_model(checkpoint, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format") != "bridge-st-v1":
        raise ValueError("Use convert-checkpoint to import a research .pth with an explicit gene panel.")
    config = Config.from_dict(saved["config"])
    model = BridgeNetwork(config, len(saved["genes"]), saved["n_features"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device).eval()
    return model, saved


def predict(data_path, checkpoint, output, device="cpu", split="all"):
    # Inference loads observations, morphology, and coordinates, never target labels.
    bundle = load_bundle(data_path, targets=False)
    device = choose_device(device)
    model, metadata = load_model(checkpoint, device)
    if bundle.genes.tolist() != metadata["genes"]:
        raise ValueError("The input gene names/order do not match the checkpoint.")
    features = bundle.node_features
    if features.shape[1] != metadata["n_features"]:
        raise ValueError("Image feature width does not match the checkpoint.")
    selected = np.ones(len(bundle.cell_ids), dtype=bool)
    if split == "test":
        selected = bundle.test_spot_mask[bundle.cell_spot_index]
    elif split != "all":
        raise ValueError("Prediction split must be all or test.")
    predictions = np.zeros((len(bundle.cell_ids), len(bundle.genes)), dtype=np.float32)
    with torch.no_grad():
        for spot, cells in enumerate(bundle.groups):
            if not len(cells) or not selected[cells].any():
                continue
            edges = torch.as_tensor(bundle.local_edges(spot, model.config.edge_radius), dtype=torch.long, device=device)
            pred, _, _ = model.forward_group(torch.as_tensor(features[cells], device=device), torch.as_tensor(bundle.spot_expression[spot], dtype=torch.float32, device=device), edges)
            predictions[cells] = pred.cpu().numpy()
    if not selected.any():
        raise ValueError("The requested prediction split contains no cells.")
    if not np.isfinite(predictions[selected]).all():
        raise RuntimeError("Model produced non-finite predictions.")
    output = Path(output)
    if output.suffix != ".npz":
        raise ValueError("Prediction output must use the .npz extension.")
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, predictions=predictions[selected], cell_ids=bundle.cell_ids[selected], genes=bundle.genes, coordinates=bundle.coordinates[selected], spot_ids=bundle.spot_ids[bundle.cell_spot_index[selected]])
    return output


def convert_checkpoint(source, gene_file, config, output):
    """Import tensor-only research state dicts without executable pickle objects."""
    old = torch.load(source, map_location="cpu", weights_only=True)
    genes = [x.strip() for x in Path(gene_file).read_text().splitlines() if x.strip()]
    if len(genes) != len(set(genes)) or not genes:
        raise ValueError("The checkpoint gene panel must contain unique names in training order.")
    encoder = old["normal_encoder_state_dict"]
    width = encoder["fc1.weight"].shape[1]
    hidden = encoder["fc2.weight"].shape[0]
    if config.hidden_dim != hidden:
        raise ValueError("hidden_dim does not match the research checkpoint.")
    model = BridgeNetwork(config, len(genes), width)
    model.normal_encoder.load_state_dict(encoder)
    if config.mode == "hierarchical":
        model.graph.load_state_dict(old["model_state_dict"])
        model.background_predictor.load_state_dict(old["noise_predictor_state_dict"])
    else:
        model.refiner.load_state_dict(old["refiner_state_dict"])
        for active, field, key in ((config.use_pos_head, "pos_head", "pos_head_state_dict"), (config.use_node_gnn, "node_gnn", "node_gnn_state_dict"), (config.expr_head, "expr_head", "expr_head_state_dict")):
            if active:
                state = old[key]
                if field == "expr_head":
                    state = {k.removeprefix("head."): v for k, v in state.items()}
                getattr(model, field).load_state_dict(state, strict=True)
        for key, name in (("expr_gate", "expr_gate"), ("expr_gate_hard", "expr_gate_hard"), ("expr_gate_thr", "expr_gate_threshold")):
            if key in old and getattr(config, name) != old[key]:
                raise ValueError(f"Configuration {name} disagrees with checkpoint metadata.")
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "bridge-st-v1", "version": __version__, "config": config.to_dict(), "genes": genes, "n_features": int(width), "state_dict": model.state_dict()}, output)
    return output
