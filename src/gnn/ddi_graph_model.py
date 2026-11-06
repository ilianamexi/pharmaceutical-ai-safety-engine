"""Graph Convolutional Network for drug-drug interaction (DDI) link prediction.

The BioSNAP ChCh-Miner graph has no node attributes, so every drug gets a learnable
embedding that two GCN layers refine with neighbourhood information. An edge score
is the dot product of the two drug embeddings (a standard GAE-style decoder).
The model is transductive: it can only score drugs that are nodes of the graph.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DDIGraphModel(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int = 64, hidden_channels: int = 64,
                 out_channels: int = 32, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.embedding.weight
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

    @staticmethod
    def decode(z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """pairs: LongTensor of shape [2, n] -> logits of shape [n]."""
        return (z[pairs[0]] * z[pairs[1]]).sum(dim=-1)

    def forward(self, edge_index: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(edge_index), pairs)
