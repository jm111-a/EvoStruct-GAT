import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm


class PPI_GAT_DualChain(nn.Module):
    """
    【架构消融版】GAT 模型 (w/o Multi-scale Fusion & w/o Skip Connections)
    此版本移除了残差跳连和多尺度特征拼接，用于评估这两项设计的必要性。
    """

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

        # 消融修改：由于不再拼接 3 种尺度的特征，输入维度改为单层的 hidden_dim
        self.final_fuse = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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

        # 特征解耦
        x_esm = x[:, :self.esm_dim]
        x_struct = x[:, self.esm_dim:]

        # 独立特征提取
        esm_emb = self.esm_proj(x_esm)
        struct_emb = self.struct_proj(x_struct)

        # 特征拼接融合
        h_concat = torch.cat([esm_emb, struct_emb], dim=-1)
        h = self.initial_fuse(h_concat)

        # 第一层 GAT
        h1 = self.gat1(h, edge_index, edge_attr)
        h1 = self.norm1(h1)
        # 消融修改：移除残差跳连 (删掉了 + h)
        h1 = F.silu(h1)

        # 第二层 GAT
        h2 = self.gat2(h1, edge_index, edge_attr)
        h2 = self.norm2(h2)
        # 消融修改：移除残差跳连 (删掉了 + h1)
        h2 = F.silu(h2)

        # 消融修改：移除多尺度特征聚合 (直接使用最后一层的输出)
        out = self.final_fuse(h2)

        return self.classifier(out).view(-1)