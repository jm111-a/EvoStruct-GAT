import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm


class EvoStruct_GAT(nn.Module):

    def __init__(self, in_dim=1284, edge_dim=5, esm_dim=1280, hidden_dim=256, heads=4, dropout=0.5):
        super().__init__()
        self.esm_dim = esm_dim
        self.struct_dim = in_dim - esm_dim
        self.dropout = dropout

        self.esm_proj = nn.Sequential(
            nn.Linear(self.esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.struct_proj = nn.Sequential(
            nn.Linear(self.struct_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.initial_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.norm1 = LayerNorm(hidden_dim)

        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.norm2 = LayerNorm(hidden_dim)

        self.final_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x_esm = x[:, :self.esm_dim]
        x_struct = x[:, self.esm_dim:]

        esm_emb = self.esm_proj(x_esm)
        struct_emb = self.struct_proj(x_struct)

        h_concat = torch.cat([esm_emb, struct_emb], dim=-1)
        h = self.initial_fuse(h_concat)

        h1 = self.gat1(h, edge_index, edge_attr)
        h1 = self.norm1(h1)
        h1 = F.silu(h1 + h)

        h2 = self.gat2(h1, edge_index, edge_attr)
        h2 = self.norm2(h2)
        h2 = F.silu(h2 + h1)

        combined = torch.cat([esm_emb, h1, h2], dim=-1)

        out = self.final_fuse(combined)
        return self.classifier(out).view(-1)