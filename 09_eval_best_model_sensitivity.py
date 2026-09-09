import os
import torch
import importlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
# 【修改】补充导入了画 ROC 和 PR 曲线所需的函数
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from torch_geometric.loader import DataLoader

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 1. 配置项 ======================
MODEL_PATH = None
OUTPUT_DIR = None

THRESH_MIN = 0.05
THRESH_MAX = 0.95
SEARCH_STEP = 91
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 2. 动态导入数据与模型 ======================
print("📦 正在加载数据集与模型架构...")
try:
    dataset_module = importlib.import_module("06_dataset_sensitivity")
    val_dataset = dataset_module.val_dataset
    test_dataset = dataset_module.test_dataset

    MODEL_PATH = os.path.join(
        dataset_module.MODEL_SAVE_DIR,
        "best_model.pth"
    )

    OUTPUT_DIR = dataset_module.MODEL_SAVE_DIR

    model_module = importlib.import_module("07_model")
    PPI_Model = getattr(model_module, "PPI_GAT_DualChain", getattr(model_module, "PPI_GAT_MultiModal", None))
except Exception as e:
    print(f"❌ 模块加载失败: {e}")
    exit()

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("🔍 加载最优模型权重...")
# 注意：这里需要传入模型初始化所需的 in_dim 和 edge_dim (从数据中动态获取最安全)
sample_data = val_dataset[0]
in_dim = sample_data.x.shape[1]
edge_dim = sample_data.edge_attr.shape[1] if sample_data.edge_attr is not None else 0

model = PPI_Model(in_dim=in_dim, edge_dim=edge_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ 模型加载完成！")


# ====================== 核心推理函数 ======================
@torch.no_grad()
def get_predictions(loader, desc="推理中"):
    all_probs = []
    all_labels = []
    for batch in tqdm(loader, desc=desc):
        batch = batch.to(DEVICE)
        out = model(batch).view(-1)
        # 🌟 修复关键Bug: 必须加 Sigmoid 将 Logits 转为 0~1 的概率
        probs = torch.sigmoid(out)
        all_probs.append(probs)
        all_labels.append(batch.y.view(-1))

    return torch.cat(all_probs).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


# ====================== 3. 第一步：在验证集上找最优阈值 ======================
print("\n📊 [阶段 1] 正在验证集上进行全局搜索最优阈值...")
val_probs, val_labels = get_predictions(val_loader, desc="验证集预测(Batch)")

best_f1 = 0
best_thresh = 0.5
best_metrics = {}

# 遍历寻找最优阈值
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

# ====================== 4. 第二步：用最优阈值在测试集上做最终评估 ======================
print(f"\n🧪 [阶段 2] 正在盲测！使用固定的门槛值 {best_thresh:.3f} 评估测试集...")
test_probs, test_labels = get_predictions(test_loader, desc="测试集预测(Batch)")

# 使用从验证集得来的 best_thresh 进行二值化
test_pred_binary = (test_probs > best_thresh).astype(int)

# 计算测试集最终指标
test_auroc = roc_auc_score(test_labels, test_probs)
test_precision = precision_score(test_labels, test_pred_binary, zero_division=0)
test_recall = recall_score(test_labels, test_pred_binary, zero_division=0)
test_f1 = f1_score(test_labels, test_pred_binary, zero_division=0)
test_accuracy = accuracy_score(test_labels, test_pred_binary)

print("\n" + "=" * 50)
print("🏆 测试集最终结果（完全无数据泄露的真实水平）：")
print(f"   执行固定阈值: {best_thresh:.3f}")
print(f"   最终 AUROC:    {test_auroc:.4f}")
print(f"   最终 F1-Score: {test_f1:.4f}")
print(f"   最终 Precision: {test_precision:.4f}")
print(f"   最终 Recall:    {test_recall:.4f}")
print(f"   最终 Accuracy:  {test_accuracy:.4f}")
print("=" * 50)


# ====================== 5. 核心重构：学术规范级可视化 ======================
print("\n🎨 正在生成学术级测试集评估面板 (包含 ROC、PR、混淆矩阵与指标表)...")

# 注入高信噪比学术画图风格
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')  # 兼容低版本

plt.rcParams.update({
    'font.family': 'Arial',       # 国际标准无衬线字体
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,       # 加粗曲线
    'figure.dpi': 300             # 确保高清输出
})

# 创建 2x2 拼图画布
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ----------------- 子图 1: ROC 曲线 -----------------
fpr, tpr, roc_thresholds = roc_curve(test_labels, test_probs)
roc_auc = auc(fpr, tpr)
axes[0, 0].plot(fpr, tpr, color='#1f77b4', label=f'ROC Curve (AUC = {roc_auc:.4f})')
axes[0, 0].plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.7)

# 标注最优阈值点
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

# ----------------- 子图 2: PR 曲线 -----------------
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(test_labels, test_probs)
pr_auc = average_precision_score(test_labels, test_probs)
axes[0, 1].plot(recall_curve, precision_curve, color='#ff7f0e', label=f'PR Curve (AUPRC = {pr_auc:.4f})')

baseline = sum(test_labels) / len(test_labels)
axes[0, 1].axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, label=f'Baseline ({baseline:.3f})')

# 标注最优阈值点 (注意 pr_thresholds 比 precision/recall 少一个元素)
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

# ----------------- 子图 3: 混淆矩阵 -----------------
cm = confusion_matrix(test_labels, test_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-binding (0)', 'Binding (1)'],
            yticklabels=['Non-binding (0)', 'Binding (1)'],
            annot_kws={"size": 16, "weight": "bold"}, ax=axes[1, 0])
axes[1, 0].set_xlabel('Predicted Label', fontweight='bold', labelpad=10)
axes[1, 0].set_ylabel('True Label', fontweight='bold', labelpad=10)
axes[1, 0].set_title(f'Test Set Confusion Matrix\n(Threshold = {best_thresh:.2f})', pad=15)

# ----------------- 子图 4: 指标表格 -----------------
axes[1, 1].axis('off')  # 隐藏坐标轴
axes[1, 1].set_title('Test Set Metrics Summary', pad=20, fontweight='bold')

metrics_data = [
    ['AUROC', f"{test_auroc:.4f}"],
    ['F1-Score', f"{test_f1:.4f}"],
    ['Precision', f"{test_precision:.4f}"],
    ['Recall', f"{test_recall:.4f}"],
    ['Accuracy', f"{test_accuracy:.4f}"]
]

# 绘制表格
table = axes[1, 1].table(cellText=metrics_data,
                         colLabels=['Metric', 'Value'],
                         loc='center',
                         cellLoc='center',
                         bbox=[0.15, 0.1, 0.7, 0.8]) # 控制表格大小和位置
table.auto_set_font_size(False)
table.set_fontsize(14)

# 表格高级美化 (斑马线与表头颜色分配)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('white')  # 隐藏内部细黑线，利用背景色做隔离
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4c72b0')  # 经典的沉稳学术蓝表头
    else:
        cell.set_text_props(weight='bold' if col == 1 else 'normal')
        if row % 2 == 0:
            cell.set_facecolor('#f2f2f2')  # 浅灰交替
        else:
            cell.set_facecolor('#e6e6e6')

# 统筹排版与保存
plt.tight_layout(pad=3.0)
final_plot_path = os.path.join(OUTPUT_DIR, "academic_evaluation_dashboard.png")
plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
print(f"✅ 包含图表与矩阵的学术面板已生成并保存至: {final_plot_path}")