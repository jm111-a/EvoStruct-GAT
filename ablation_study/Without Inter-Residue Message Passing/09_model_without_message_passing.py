import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm


ABLATION_NAME = "without_inter_residue_message_passing"


class PPI_GAT_DualChain(nn.Module):
    """
    单因素消融：w/o inter-residue message passing。
    保留节点特征投射、初始融合、最终融合和分类头，但绕过两层 GATv2，
    用于评价跨残基图消息传递的整体贡献。
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
        esm_emb = self.esm_proj(x_esm)
        struct_emb = self.struct_proj(x_struct)

        # 特征拼接融合 (取代原先的相加)
        h_concat = torch.cat([esm_emb, struct_emb], dim=-1)
        h = self.initial_fuse(h_concat)

        # 单因素消融：绕过两层 GATv2，不进行跨残基消息传递。
        # h 保留每个节点自身的 ESM-2 与 DSSP 融合信息。
        # 第三个位置使用零张量占位，以保持 final_fuse 的输入维度不变。
        zero_graph = torch.zeros_like(h)
        combined = torch.cat([esm_emb, h, zero_graph], dim=-1)

        # 降维并输出
        out = self.final_fuse(combined)
        return self.classifier(out).view(-1)