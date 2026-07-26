import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm


class PPI_GAT_DualChain(nn.Module):
    """
    【单因素架构消融】GAT 模型 (w/o Multiscale Fusion)
    此版本仅移除多尺度特征拼接，并完整保留两层残差跳连。
    """

    def __init__(self, in_dim=1284, edge_dim=5, esm_dim=1280, hidden_dim=256, heads=4, dropout=0.5):
        super().__init__()
        self.esm_dim = esm_dim
        self.struct_dim = in_dim - esm_dim
        self.dropout = dropout

        # 1. ESM 特征独立投射
        self.esm_proj = nn.Sequential(
            nn.Linear(self.esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # 2. 结构特征独立投射
        self.struct_proj = nn.Sequential(
            nn.Linear(self.struct_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # 3. 初始特征拼接降维层
        self.initial_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        # 4. 多头注意力层 (GATv2)
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.norm1 = LayerNorm(hidden_dim)

        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.norm2 = LayerNorm(hidden_dim)

        # 5. 最终融合层
        # 单因素消融：不再拼接多尺度特征，因此输入维度为 hidden_dim
        self.final_fuse = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # 6. 最终分类头
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
        # 保留第一层残差跳连
        h1 = F.silu(h1 + h)

        # 第二层 GAT
        h2 = self.gat2(h1, edge_index, edge_attr)
        h2 = self.norm2(h2)
        # 保留第二层残差跳连
        h2 = F.silu(h2 + h1)

        # 单因素消融：移除多尺度特征聚合，仅使用最后一层 GAT 表示
        out = self.final_fuse(h2)

        return self.classifier(out).view(-1)