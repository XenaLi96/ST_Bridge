#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.nn import GCNConv
except Exception:  # pragma: no cover - optional dependency in some envs
    GCNConv = None


class ResidualRefiner(nn.Module):
    """Refine super-node coarse signal to normal-node prediction via residual."""

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, node_hidden: torch.Tensor, coarse_gene: torch.Tensor) -> torch.Tensor:
        # node_hidden: [N, H]; coarse_gene: [N, G] (broadcasted super-node gene vector)
        residual = self.refine(node_hidden)
        return F.relu(coarse_gene + residual)


class ExprHead(nn.Module):
    """Predict per-gene expression logits from node hidden features."""

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, node_hidden: torch.Tensor) -> torch.Tensor:
        return self.head(node_hidden)


class PosHead(nn.Module):
    """Predict positive-valued expression residuals from node hidden features."""

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, node_hidden: torch.Tensor) -> torch.Tensor:
        return self.head(node_hidden)


class NodeGNN(nn.Module):
    """Optional node-level GNN to refine hidden features within a super-node."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        if GCNConv is None:
            self.conv1 = None
            self.conv2 = None
        else:
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.conv1 is None or edge_index is None or edge_index.numel() == 0:
            return x
        x = self.conv1(x, edge_index)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
