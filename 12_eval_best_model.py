import os
import torch
import importlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from torch_geometric.loader import DataLoader

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

MODEL_PATH = r"C:\Users\Administrator\Desktop\P-P\model_save\best_model.pth"
OUTPUT_DIR = r"C:\Users\Administrator\Desktop\P-P\model_save"

THRESH_MIN = 0.05
THRESH_MAX = 0.95
SEARCH_STEP = 91
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("📦 正在加载数据集与模型架构...")
try:
    dataset_module = importlib.import_module("08_dataset")
    val_dataset = dataset_module.val_dataset
    test_dataset = dataset_module.test_dataset

    model_module = importlib.import_module("09_model")
    PPI_Model = getattr(model_module, "EvoStruct_GAT", None)
except Exception as e:
    print(f"❌ 模块加载失败: {e}")
    exit()

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("🔍 加载最优模型权重...")
sample_data = val_dataset[0]
in_dim = sample_data.x.shape[1]
edge_dim = sample_data.edge_attr.shape[1] if sample_data.edge_attr is not None else 0

model = PPI_Model(in_dim=in_dim, edge_dim=edge_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ 模型加载完成！")


@torch.no_grad()
def get_predictions(loader, desc="推理中"):
    all_probs = []
    all_labels = []
    for batch in tqdm(loader, desc=desc):
        batch = batch.to(DEVICE)
        out = model(batch).view(-1)
        probs = torch.sigmoid(out)
        all_probs.append(probs)
        all_labels.append(batch.y.view(-1))

    return torch.cat(all_probs).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


print("\n📊 [阶段 1] 正在验证集上进行全局搜索最优阈值...")
val_probs, val_labels = get_predictions(val_loader, desc="验证集预测(Batch)")

best_f1 = 0
best_thresh = 0.5
best_metrics = {}

for thresh in np.linspace(THRESH_MIN, THRESH_MAX, SEARCH_STEP):
    pred_binary = (val_probs > thresh).astype(int)
    f1 = f1_score(val_labels, pred_binary, zero_division=0)

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh
        best_metrics = {
            "precision": precision_score(val_labels, pred_binary, zero_division=0),
            "recall": recall_score(val_labels, pred_binary, zero_division=0),
            "f1": f1,
            "accuracy": accuracy_score(val_labels, pred_binary),
            "auroc": roc_auc_score(val_labels, val_probs)
        }

print("\n🎉 验证集寻优结束：")
print(f"   锁定最优分类阈值: {best_thresh:.3f}")
print(f"   验证集极限 F1: {best_metrics['f1']:.4f} (AUC: {best_metrics['auroc']:.4f})")

print(f"\n🧪 [阶段 2] 正在盲测！使用固定的门槛值 {best_thresh:.3f} 评估测试集...")
test_probs, test_labels = get_predictions(test_loader, desc="测试集预测(Batch)")

test_pred_binary = (test_probs > best_thresh).astype(int)

test_auroc = roc_auc_score(test_labels, test_probs)
test_precision = precision_score(test_labels, test_pred_binary, zero_division=0)
test_recall = recall_score(test_labels, test_pred_binary, zero_division=0)
test_f1 = f1_score(test_labels, test_pred_binary, zero_division=0)
test_accuracy = accuracy_score(test_labels, test_pred_binary)

print("\n" + "=" * 50)
print("🏆 测试集最终结果：")
print(f"   执行固定阈值: {best_thresh:.3f}")
print(f"   最终 AUROC:    {test_auroc:.4f}")
print(f"   最终 F1-Score: {test_f1:.4f}")
print(f"   最终 Precision: {test_precision:.4f}")
print(f"   最终 Recall:    {test_recall:.4f}")
print(f"   最终 Accuracy:  {test_accuracy:.4f}")
print("=" * 50)


print("\n🎨 正在生成学术级测试集评估面板 (包含 ROC、PR、混淆矩阵与指标表)...")

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


fig, axes = plt.subplots(2, 2, figsize=(14, 12))

fpr, tpr, roc_thresholds = roc_curve(test_labels, test_probs)
roc_auc = auc(fpr, tpr)
axes[0, 0].plot(fpr, tpr, color='#1f77b4', label=f'ROC Curve (AUC = {roc_auc:.4f})')
axes[0, 0].plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.7)

best_idx_roc = np.argmin(np.abs(roc_thresholds - best_thresh))
axes[0, 0].plot(fpr[best_idx_roc], tpr[best_idx_roc], marker='*', markersize=15, color='#d62728', markeredgecolor='white', label=f'Optimal Thresh ({best_thresh:.2f})', zorder=5)

axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('Receiver Operating Characteristic')
axes[0, 0].legend(loc="lower right", frameon=True, edgecolor='black')
axes[0, 0].spines['top'].set_visible(False)
axes[0, 0].spines['right'].set_visible(False)
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.6)
axes[0, 0].grid(axis='x', visible=False)

precision_curve, recall_curve, pr_thresholds = precision_recall_curve(test_labels, test_probs)
pr_auc = average_precision_score(test_labels, test_probs)
axes[0, 1].plot(recall_curve, precision_curve, color='#ff7f0e', label=f'PR Curve (AUPRC = {pr_auc:.4f})')

baseline = sum(test_labels) / len(test_labels)
axes[0, 1].axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, label=f'Baseline ({baseline:.3f})')

best_idx_pr = np.argmin(np.abs(pr_thresholds - best_thresh))
axes[0, 1].plot(recall_curve[best_idx_pr], precision_curve[best_idx_pr], marker='*', markersize=15, color='#d62728', markeredgecolor='white', label=f'Optimal Thresh ({best_thresh:.2f})', zorder=5)

axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall Curve')
axes[0, 1].legend(loc="lower left", frameon=True, edgecolor='black')
axes[0, 1].spines['top'].set_visible(False)
axes[0, 1].spines['right'].set_visible(False)
axes[0, 1].grid(axis='y', linestyle='--', alpha=0.6)
axes[0, 1].grid(axis='x', visible=False)

cm = confusion_matrix(test_labels, test_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-binding (0)', 'Binding (1)'],
            yticklabels=['Non-binding (0)', 'Binding (1)'],
            annot_kws={"size": 16, "weight": "bold"}, ax=axes[1, 0])
axes[1, 0].set_xlabel('Predicted Label', fontweight='bold', labelpad=10)
axes[1, 0].set_ylabel('True Label', fontweight='bold', labelpad=10)
axes[1, 0].set_title(f'Test Set Confusion Matrix\n(Threshold = {best_thresh:.2f})', pad=15)

axes[1, 1].axis('off')
axes[1, 1].set_title('Test Set Metrics Summary', pad=20, fontweight='bold')

metrics_data = [
    ['AUROC', f"{test_auroc:.4f}"],
    ['F1-Score', f"{test_f1:.4f}"],
    ['Precision', f"{test_precision:.4f}"],
    ['Recall', f"{test_recall:.4f}"],
    ['Accuracy', f"{test_accuracy:.4f}"]
]

table = axes[1, 1].table(cellText=metrics_data,
                         colLabels=['Metric', 'Value'],
                         loc='center',
                         cellLoc='center',
                         bbox=[0.15, 0.1, 0.7, 0.8])
table.auto_set_font_size(False)
table.set_fontsize(14)


for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('white')
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4c72b0')
    else:
        cell.set_text_props(weight='bold' if col == 1 else 'normal')
        if row % 2 == 0:
            cell.set_facecolor('#f2f2f2')
        else:
            cell.set_facecolor('#e6e6e6')

plt.tight_layout(pad=3.0)
final_plot_path = os.path.join(OUTPUT_DIR, "academic_evaluation_dashboard.png")
plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
print(f"✅ 包含图表与矩阵的学术面板已生成并保存至: {final_plot_path}")