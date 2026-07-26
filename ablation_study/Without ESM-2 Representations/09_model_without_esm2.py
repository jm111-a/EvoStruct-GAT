import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm


ABLATION_NAME = "without_esm2_representations"


class PPI_GAT_DualChain(nn.Module):
    """
    单因素消融：w/o ESM-2 representations。
    仅移除 ESM-2 序列表征；DSSP、图拓扑、边属性、GATv2、残差连接和多尺度融合均保留。
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

        # 3. 初始特征拼接降维层 (将 esm 和 struct 拼接后降回 hidden_dim)
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

        # 5. 多尺度最终融合层 (融合 原始ESM + GAT1输出 + GAT2输出)
        self.final_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
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
        struct_emb = self.struct_proj(x_struct)

        # 单因素消融：完全屏蔽 ESM-2 信息。
        # 使用与 struct_emb 相同形状的零张量，避免 Linear bias 产生伪序列信号，
        # 同时保持后续网络结构和张量维度不变。
        esm_emb = torch.zeros_like(struct_emb)

        # 特征拼接融合 (取代原先的相加)
        h_concat = torch.cat([esm_emb, struct_emb], dim=-1)
        h = self.initial_fuse(h_concat)

        # 第一层 GAT + 残差跳连
        h1 = self.gat1(h, edge_index, edge_attr)
        h1 = self.norm1(h1)
        h1 = F.silu(h1 + h)  # 残差连接，保留初始节点特征

        # 第二层 GAT + 残差跳连
        h2 = self.gat2(h1, edge_index, edge_attr)
        h2 = self.norm2(h2)
        h2 = F.silu(h2 + h1)  # 残差连接，保留一层局部特征

        # 多尺度特征聚合 (Multi-scale Fusion)
        # 将 "纯序列特征"、"一层邻居特征" 和 "二层全局特征" 拼接到一起
        combined = torch.cat([esm_emb, h1, h2], dim=-1)

        # 降维并输出
        out = self.final_fuse(combined)
        return self.classifier(out).view(-1)