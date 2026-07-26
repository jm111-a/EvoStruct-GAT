import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# ====================== 1. 环境与模块加载 ======================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

try:
    dataset_module = importlib.import_module("08_dataset")
    model_module = importlib.import_module("09_model")

    train_dataset = dataset_module.train_dataset
    val_dataset = dataset_module.val_dataset
    test_dataset = dataset_module.test_dataset
    MODEL_SAVE_DIR = dataset_module.MODEL_SAVE_DIR
    PPI_Model = getattr(model_module, "EvoStruct_GAT", None)
except Exception as e:
    print(f"❌ 模块加载失败: {e}")
    exit()


# ====================== 2. Focal Loss 定义 (保留类定义，但下方不再调用) ======================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()


# ====================== 3. 参数设置 (数据平衡优化版) ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
EPOCHS = 100
PATIENCE = 20
UNDERSAMPLE_RATIO = 3

# --- A. 图级界面增强采样 (WeightedRandomSampler) ---
print("📊 正在计算样本权重以实施界面增强采样...")
train_weights = []
for data in train_dataset:
    pos_count = (data.y == 1).sum().item()
    total_nodes = data.y.numel()
    weight = (pos_count + 1) / total_nodes
    train_weights.append(weight)

sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

# 加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = PPI_Model(in_dim=train_dataset[0].x.shape[1],
                  edge_dim=train_dataset[0].edge_attr.shape[1] if train_dataset[0].edge_attr is not None else 0).to(
    DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# 🌟【策略消融修改 1】使用普通的 BCE Loss，移除 Focal Loss
criterion = nn.BCEWithLogitsLoss()


# ====================== 4. 指标计算工具 ======================
@torch.no_grad()
def get_metrics(loader, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0
    for data in loader:
        data = data.to(DEVICE)
        out = model(data)
        total_loss += criterion(out, data.y.float()).item()
        all_probs.append(torch.sigmoid(out))
        all_labels.append(data.y)

    y_true = torch.cat(all_labels).cpu().numpy()
    y_score = torch.cat(all_probs).cpu().numpy()
    y_pred = (y_score > threshold).astype(int)

    return {
        'loss': total_loss / len(loader),
        'auroc': roc_auc_score(y_true, y_score),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred)
    }, y_score, y_true


# ====================== 5. 训练循环 ======================
print(f"🚀 开始策略消融训练... (w/o Focal Loss & w/o Undersampling)")
best_val_f1 = 0
patience_counter = 0
history = []
best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model_ablation.pth")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for data in tqdm(train_loader, desc=f"Epoch {epoch:03d}/{EPOCHS}"):
        data = data.to(DEVICE)
        optimizer.zero_grad()

        out = model(data)

        # 🌟【策略消融修改 2】关闭节点级动态下采样，直接在全图节点上计算 Loss
        loss = criterion(out, data.y.float())

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    train_m, _, _ = get_metrics(train_loader)
    val_m, _, _ = get_metrics(val_loader)
    test_m, _, _ = get_metrics(test_loader)
    history.append({'epoch': epoch, 'train': train_m, 'val': val_m, 'test': test_m})

    print(f"-> Ep {epoch:03d} | Val F1: {val_m['f1']:.4f} | Val Loss: {val_m['loss']:.4f}")

    # --- C. 早停逻辑 ---
    if val_m['f1'] > best_val_f1:
        best_val_f1 = val_m['f1']
        torch.save(model.state_dict(), best_model_path)
        patience_counter = 0
        print(f"🌟 性能提升，模型已保存！")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"🛑 连续 {PATIENCE} 轮无提升，触发早停。")
            break

# ====================== 6. 评估与可视化 ======================
print("\n🧪 正在加载最佳模型进行最终评估...")
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))

_, val_probs, val_labels = get_metrics(val_loader)
_, test_probs, test_labels = get_metrics(test_loader)

# 最优阈值搜索
best_t = 0.5
max_f1 = 0
for t in np.arange(0.1, 0.8, 0.01):
    f1 = f1_score(val_labels, (val_probs > t).astype(int), zero_division=0)
    if f1 > max_f1:
        max_f1 = f1
        best_t = t

y_pred_test = (test_probs > best_t).astype(int)
print("\n" + "=" * 40)
print(f"🎯 最终测试集指标 (最优阈值: {best_t:.2f}):")
print(f"   - AUROC: {roc_auc_score(test_labels, test_probs):.4f}")
print(f"   - F1-Score: {f1_score(test_labels, y_pred_test, zero_division=0):.4f}")
print(f"   - Precision: {precision_score(test_labels, y_pred_test, zero_division=0):.4f}")
print(f"   - Recall: {recall_score(test_labels, y_pred_test, zero_division=0):.4f}")
print(f"   - Accuracy: {accuracy_score(test_labels, y_pred_test):.4f}")
print("=" * 40 + "\n")

# ====================== 核心重构：学术规范级绘图逻辑 ======================
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,
    'figure.dpi': 300
})

df_plot = pd.DataFrame([{'epoch': e['epoch'], 'split': s, **e[s]} for e in history for s in ['train', 'val']])

val_df = df_plot[df_plot['split'] == 'val'].reset_index(drop=True)
best_idx = val_df['f1'].idxmax()
best_epoch = int(val_df.loc[best_idx, 'epoch'])
best_val_f1_score = val_df.loc[best_idx, 'f1']

metrics_to_plot = ['loss', 'auroc', 'f1']
titles = ['Loss Curve', 'AUROC Curve', 'F1-Score Curve']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = {'train': '#1f77b4', 'val': '#ff7f0e'}

for i, m in enumerate(metrics_to_plot):
    ax = axes[i]
    for s in ['train', 'val']:
        d = df_plot[df_plot['split'] == s]
        ax.plot(d['epoch'], d[m], label=f"{s.capitalize()} Set", color=colors[s], alpha=0.9)

    ax.set_title(titles[i], pad=15)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel(m.upper() if m != 'loss' else 'Loss', fontweight='bold')

    ax.axvline(x=best_epoch, color='#d62728', linestyle='--', linewidth=2, alpha=0.8,
               label=f'Best Epoch ({best_epoch})')

    if m == 'f1':
        ax.plot(best_epoch, best_val_f1_score, marker='*', markersize=15, color='#d62728', markeredgecolor='white',
                zorder=5)

    ax.legend(loc='best', frameon=True, edgecolor='black')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.grid(axis='x', visible=False)

plt.tight_layout()
save_path = os.path.join(MODEL_SAVE_DIR, 'academic_training_curves_ablation.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f"✅ 学术级规范画图完成！结果已存至: {save_path}")