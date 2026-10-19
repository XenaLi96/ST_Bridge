"""Composition of the original BRIDGE and C2F model definitions."""

from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import GraphAutoencoder, NoisePredictor, NormalEncoder
from .refinement import ResidualRefiner, NodeGNN, PosHead


@dataclass
class Config:
    mode: str = "c2f"
    hidden_dim: int = 64
    epochs: int = 12
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 32
    seed: int = 0
    edge_radius: float = 150.0
    use_node_gnn: bool = False
    use_pos_head: bool = False
    expr_head: bool = False
    expr_gate: bool = False
    expr_gate_hard: bool = False
    expr_gate_threshold: float = 0.5
    expression_threshold: float = 0.3
    expr_lambda: float = 0.2
    rate_lambda: float = 0.0
    consistency_scale: float = 1.0
    use_sum: bool = False
    node_loss_weight: float = 1.0
    l1_lambda: float = 0.0
    variance_lambda: float = 0.0
    quantile_lambda: float = 0.0
    quantiles: tuple = (0.1, 0.5, 0.9)
    celltype: bool = False
    celltype_k: int = 8
    celltype_lambda: float = 0.2
    celltype_sample: int = 200000
    rank_lambda: float = 0.0
    rank_pairs: int = 256
    rank_epsilon: float = 0.001
    rank_tau: float = 1.0
    marker_sets: object = None
    marker_lambda: float = 0.5
    marker_kl_lambda: float = 0.2
    marker_ce_lambda: float = 0.2
    marker_temperature: float = 1.0

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        cfg = cls(**values)
        if cfg.mode not in {"c2f", "hierarchical"}:
            raise ValueError("mode must be c2f or hierarchical")
        for key in ("hidden_dim", "epochs", "batch_size", "celltype_k", "rank_pairs"):
            if not isinstance(getattr(cfg, key), int) or getattr(cfg, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("learning_rate", "edge_radius", "rank_tau", "marker_temperature"):
            if getattr(cfg, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if cfg.celltype_sample < 0:
            raise ValueError("celltype_sample must be nonnegative")
        for key in ("weight_decay", "expr_lambda", "rate_lambda", "consistency_scale", "node_loss_weight", "l1_lambda", "variance_lambda", "quantile_lambda", "celltype_lambda", "rank_lambda", "rank_epsilon", "marker_lambda", "marker_kl_lambda", "marker_ce_lambda"):
            if getattr(cfg, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if not 0 <= cfg.expr_gate_threshold <= 1 or not cfg.quantiles or any(not 0 <= q <= 1 for q in cfg.quantiles):
            raise ValueError("Gate threshold and quantiles must be in [0, 1].")
        if cfg.expr_gate and not cfg.expr_head:
            raise ValueError("expr_gate requires expr_head")
        if cfg.rate_lambda and not cfg.expr_head:
            raise ValueError("rate_lambda requires expr_head")
        if cfg.mode == "hierarchical" and (cfg.expr_head or cfg.use_node_gnn or cfg.use_pos_head or cfg.celltype or cfg.marker_sets or cfg.rank_lambda or cfg.quantile_lambda or cfg.variance_lambda):
            raise ValueError("C2F auxiliary heads/losses cannot be enabled in hierarchical mode.")
        return cfg

    def to_dict(self):
        return asdict(self)


class BridgeNetwork(nn.Module):
    def __init__(self, config, n_genes, n_features):
        super().__init__()
        self.config = config
        self.normal_encoder = NormalEncoder(n_features, config.hidden_dim)
        if config.mode == "hierarchical":
            self.graph = GraphAutoencoder(n_genes, config.hidden_dim, "gcn")
            self.background_predictor = NoisePredictor(config.hidden_dim, n_genes)
        else:
            self.refiner = ResidualRefiner(config.hidden_dim, n_genes)
            if config.use_node_gnn:
                self.node_gnn = NodeGNN(config.hidden_dim)
            if config.use_pos_head:
                self.pos_head = PosHead(config.hidden_dim, n_genes)
            if config.expr_head:
                self.expr_head = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU(inplace=True), nn.Dropout(.1), nn.Linear(config.hidden_dim, n_genes))

    def forward_group(self, features, coarse, edges):
        hidden = self.normal_encoder(features)
        logits = None
        if self.config.mode == "hierarchical":
            hidden = hidden.clamp(-20, 20)
            pred = F.relu(self.graph.decoder_layer2(self.graph.decoder_layer1(hidden, edges), edges)).clamp(0, 20)
        else:
            if self.config.use_node_gnn:
                hidden = self.node_gnn(hidden, edges)
            coarse = coarse.unsqueeze(0).expand(features.shape[0], -1)
            pred = F.softplus(coarse + self.pos_head(hidden)) if self.config.use_pos_head else self.refiner(hidden, coarse)
            if self.config.expr_head:
                logits = self.expr_head(hidden)
                if self.config.expr_gate:
                    gate = torch.sigmoid(logits)
                    if self.config.expr_gate_hard:
                        hard = (gate >= self.config.expr_gate_threshold).to(gate.dtype)
                        gate = hard - gate.detach() + gate if self.training else hard
                    pred = pred * gate
        return pred, hidden, logits
