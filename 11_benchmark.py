import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import importlib
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score, roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    dataset_module = importlib.import_module("08_dataset")
    model_module = importlib.import_module("09_model")
    train_dataset = dataset_module.train_dataset
    test_dataset = dataset_module.test_dataset
    MODEL_SAVE_DIR = dataset_module.MODEL_SAVE_DIR
    Our_Model = getattr(model_module, "EvoStruct_GAT")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit()


class Feature_MLP(nn.Module):
    """特征基线：只用节点特征，不用图边"""

    def __init__(self, in_dim=1284, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        return self.mlp(data.x).view(-1)


class Standard_GCN(nn.Module):
    """标准图基线：使用最普通的图卷积，无边特征"""

    def __init__(self, in_dim=1284, hidden_dim=256):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(F.silu(self.conv1(x, edge_index)), p=0.5, training=self.training)
        x = F.dropout(F.silu(self.conv2(x, edge_index)), p=0.5, training=self.training)
        return self.classifier(x).view(-1)


def train_and_eval(model_class, model_name, epochs=30):
    print(f"⏳ 正在训练与评估: {model_name} ...")
    in_dim = train_dataset[0].x.shape[1]

    if model_name == "Ours (Multi-scale GATv2)":
        model = model_class(in_dim=in_dim, edge_dim=train_dataset[0].edge_attr.shape[1]).to(DEVICE)
    else:
        model = model_class(in_dim=in_dim).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(DEVICE))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    model.train()
    for _ in range(epochs):
        for data in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(data.to(DEVICE)), data.y.float())
            loss.backward()
            optimizer.step()

    model.eval()
    y_prob, y_true = [], []
    with torch.no_grad():
        for data in test_loader:
            y_prob.append(torch.sigmoid(model(data.to(DEVICE))).cpu().numpy())
            y_true.append(data.y.cpu().numpy())

    y_prob, y_true = np.concatenate(y_prob), np.concatenate(y_true)
    y_pred = (y_prob > 0.5).astype(int)
    return {'AUROC': roc_auc_score(y_true, y_prob), 'F1': f1_score(y_true, y_pred, zero_division=0)}


if __name__ == "__main__":
    EPOCHS = 30

    results = {
        "Feature-MLP": train_and_eval(Feature_MLP, "Feature-MLP", EPOCHS),
        "Standard GCN": train_and_eval(Standard_GCN, "Standard GCN", EPOCHS),
        "Ours (Multi-scale GATv2)": train_and_eval(Our_Model, "Ours (Multi-scale GATv2)", EPOCHS)
    }

    print("\n" + "=" * 50)
    print("🏆 最终模型性能对比 (Test Set Metrics)")
    print("=" * 50)
    for model_name, metrics in results.items():
        print(f"🔹 {model_name:<26} | AUROC: {metrics['AUROC']:.4f} | F1-Score: {metrics['F1']:.4f}")
    print("=" * 50)