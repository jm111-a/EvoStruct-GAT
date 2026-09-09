import os
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    GATv2Conv,
    LayerNorm
)

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    accuracy_score
)

from tqdm import tqdm


warnings.filterwarnings("ignore")


# ============================================================
# 0. 随机种子
# ============================================================

# 5个随机种子用于内部模型稳定性比较。
# 三个模型在每个seed下使用完全相同的seed，保证配对比较公平。
SEEDS = [42, 123, 3407, 2025, 2026]


def seed_everything(seed=42):

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False



# ============================================================
# 1. 路径
# ============================================================

ROOT = r"C:\Users\Administrator\Desktop\P-P"


GRAPH_ROOT = os.path.join(
    ROOT,
    "cluster_split_v3_A2",
    "partner_only_DSSP",
    "06_graphs"
)


TRAIN_DIR = os.path.join(
    GRAPH_ROOT,
    "train_pyg_3d_graph"
)


VAL_DIR = os.path.join(
    GRAPH_ROOT,
    "val_pyg_3d_graph"
)


TEST_DIR = os.path.join(
    GRAPH_ROOT,
    "test_pyg_3d_graph"
)


OUTPUT_DIR = os.path.join(
    ROOT,
    "cluster_split_v3_A2",
    "partner_only_DSSP",
    "07_model_save",
    "internal_comparison"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. 设备
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)

print("Internal model comparison | 5 random seeds")

print("=" * 70)

print("Device:", DEVICE)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 3. Dataset
# ============================================================

class PPIGraphDataset(Dataset):

    def __init__(
            self,
            pyg_dir,
            mode="train"
    ):

        self.pyg_dir = pyg_dir

        self.mode = mode


        if not os.path.isdir(pyg_dir):

            raise FileNotFoundError(
                f"Graph目录不存在：{pyg_dir}"
            )


        self.file_list = sorted(
            [
                f
                for f in os.listdir(pyg_dir)
                if f.endswith(".pyg")
            ]
        )


        if len(self.file_list) == 0:

            raise RuntimeError(
                f"{mode} 中没有找到 .pyg 文件：{pyg_dir}"
            )


    def __len__(self):

        return len(
            self.file_list
        )


    def __getitem__(
            self,
            idx
    ):

        file_name = (
            self.file_list[idx]
        )


        path = os.path.join(
            self.pyg_dir,
            file_name
        )


        try:

            data = torch.load(
                path,
                weights_only=False
            )

        except TypeError:

            data = torch.load(
                path
            )


        if (
            not hasattr(data, "y")
            or
            data.y is None
        ):

            raise ValueError(
                f"{file_name} 不包含 y"
            )


        if (
            not hasattr(data, "edge_attr")
            or
            data.edge_attr is None
        ):

            raise ValueError(
                f"{file_name} 不包含 edge_attr"
            )


        return data


# ============================================================
# 4. 加载当前严格split数据
# ============================================================

train_dataset = PPIGraphDataset(
    TRAIN_DIR,
    "train"
)


val_dataset = PPIGraphDataset(
    VAL_DIR,
    "val"
)


test_dataset = PPIGraphDataset(
    TEST_DIR,
    "test"
)


print("\nDataset:")

print(
    "Train graphs:",
    len(train_dataset)
)

print(
    "Val graphs:",
    len(val_dataset)
)

print(
    "Test graphs:",
    len(test_dataset)
)


sample = train_dataset[0]


print("\nGraph feature check:")

print(
    "x:",
    sample.x.shape
)

print(
    "edge_attr:",
    sample.edge_attr.shape
)

print(
    "y:",
    sample.y.shape
)


if sample.x.shape[1] != 1284:

    raise ValueError(
        f"节点特征维度异常："
        f"{sample.x.shape[1]}，预期1284"
    )


if sample.edge_attr.shape[1] != 5:

    raise ValueError(
        f"边特征维度异常："
        f"{sample.edge_attr.shape[1]}，预期5"
    )


# ============================================================
# 5. Baseline 1：Feature MLP
# ============================================================

class Feature_MLP(nn.Module):

    """
    只使用节点特征，
    不使用图拓扑。
    """

    def __init__(
            self,
            in_dim=1284,
            hidden_dim=256
    ):

        super().__init__()


        self.mlp = nn.Sequential(

            nn.Linear(
                in_dim,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.SiLU(),

            nn.Dropout(0.5),


            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),

            nn.LayerNorm(
                hidden_dim // 2
            ),

            nn.SiLU(),

            nn.Dropout(0.5),


            nn.Linear(
                hidden_dim // 2,
                1
            )
        )


    def forward(
            self,
            data
    ):

        return (
            self.mlp(
                data.x
            )
            .view(-1)
        )


# ============================================================
# 6. Baseline 2：Standard GCN
# ============================================================

class Standard_GCN(nn.Module):

    """
    标准两层GCN，
    使用graph topology，
    不使用5维edge feature。
    """

    def __init__(
            self,
            in_dim=1284,
            hidden_dim=256
    ):

        super().__init__()


        self.conv1 = GCNConv(
            in_dim,
            hidden_dim
        )


        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim // 2
        )


        self.classifier = nn.Linear(
            hidden_dim // 2,
            1
        )


    def forward(
            self,
            data
    ):

        x = data.x

        edge_index = (
            data.edge_index
        )


        x = self.conv1(
            x,
            edge_index
        )


        x = F.silu(
            x
        )


        x = F.dropout(
            x,
            p=0.5,
            training=self.training
        )


        x = self.conv2(
            x,
            edge_index
        )


        x = F.silu(
            x
        )


        x = F.dropout(
            x,
            p=0.5,
            training=self.training
        )


        return (
            self.classifier(
                x
            )
            .view(-1)
        )


# ============================================================
# 7. Ours：当前 EvoStruct-GAT
# ============================================================

class PPI_GAT_DualChain(nn.Module):

    def __init__(
            self,
            in_dim=1284,
            edge_dim=5,
            esm_dim=1280,
            hidden_dim=256,
            heads=4,
            dropout=0.5
    ):

        super().__init__()


        self.esm_dim = (
            esm_dim
        )


        self.struct_dim = (
            in_dim
            -
            esm_dim
        )


        self.dropout = (
            dropout
        )


        # ----------------------------------------------------
        # ESM2 branch
        # ----------------------------------------------------

        self.esm_proj = nn.Sequential(

            nn.Linear(
                self.esm_dim,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.SiLU(),

            nn.Dropout(
                dropout
            )
        )


        # ----------------------------------------------------
        # DSSP structural branch
        # ----------------------------------------------------

        self.struct_proj = nn.Sequential(

            nn.Linear(
                self.struct_dim,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.SiLU(),

            nn.Dropout(
                dropout
            )
        )


        # ----------------------------------------------------
        # Initial fusion
        # ----------------------------------------------------

        self.initial_fuse = nn.Sequential(

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.SiLU()
        )


        # ----------------------------------------------------
        # GATv2 1
        # ----------------------------------------------------

        self.gat1 = GATv2Conv(

            hidden_dim,

            hidden_dim // heads,

            heads=heads,

            edge_dim=edge_dim,

            dropout=dropout
        )


        self.norm1 = LayerNorm(
            hidden_dim
        )


        # ----------------------------------------------------
        # GATv2 2
        # ----------------------------------------------------

        self.gat2 = GATv2Conv(

            hidden_dim,

            hidden_dim // heads,

            heads=heads,

            edge_dim=edge_dim,

            dropout=dropout
        )


        self.norm2 = LayerNorm(
            hidden_dim
        )


        # ----------------------------------------------------
        # Multi-scale fusion
        # ----------------------------------------------------

        self.final_fuse = nn.Sequential(

            nn.Linear(
                hidden_dim * 3,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.SiLU(),

            nn.Dropout(
                dropout
            )
        )


        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),

            nn.LayerNorm(
                hidden_dim // 2
            ),

            nn.SiLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim // 2,
                1
            )
        )


    def forward(
            self,
            data
    ):

        x = data.x

        edge_index = (
            data.edge_index
        )

        edge_attr = (
            data.edge_attr
        )


        # ----------------------------------------------------
        # ESM / structural separation
        # ----------------------------------------------------

        x_esm = (
            x[
                :,
                :self.esm_dim
            ]
        )


        x_struct = (
            x[
                :,
                self.esm_dim:
            ]
        )


        # ----------------------------------------------------
        # Projection
        # ----------------------------------------------------

        esm_emb = (
            self.esm_proj(
                x_esm
            )
        )


        struct_emb = (
            self.struct_proj(
                x_struct
            )
        )


        # ----------------------------------------------------
        # Initial fusion
        # ----------------------------------------------------

        h_concat = torch.cat(
            [
                esm_emb,
                struct_emb
            ],
            dim=-1
        )


        h = self.initial_fuse(
            h_concat
        )


        # ----------------------------------------------------
        # GAT layer 1 + residual
        # ----------------------------------------------------

        h1 = self.gat1(
            h,
            edge_index,
            edge_attr
        )


        h1 = self.norm1(
            h1
        )


        h1 = F.silu(
            h1 + h
        )


        # ----------------------------------------------------
        # GAT layer 2 + residual
        # ----------------------------------------------------

        h2 = self.gat2(
            h1,
            edge_index,
            edge_attr
        )


        h2 = self.norm2(
            h2
        )


        h2 = F.silu(
            h2 + h1
        )


        # ----------------------------------------------------
        # Multi-scale
        # ----------------------------------------------------

        combined = torch.cat(

            [
                esm_emb,
                h1,
                h2
            ],

            dim=-1
        )


        out = self.final_fuse(
            combined
        )


        return (
            self.classifier(
                out
            )
            .view(-1)
        )


# ============================================================
# 8. 参数
# ============================================================

# IMPORTANT:
# 与原比较代码保持一致：
# - 模型结构不变
# - 数据目录不变
# - batch size / learning rate / weight decay / pos_weight 不变
# - 最大训练轮数仍为30
#
# 本版本保持上一版模型选择与评价流程不变，并只新增5-seed重复：
# 1. 每个epoch在validation set上计算AUROC
# 2. 依据Val AUROC保存各模型自己的最佳checkpoint
# 3. patience=10进行early stopping
# 4. 在最佳checkpoint的validation预测上选择F1最佳threshold
# 5. threshold固定后，test set只评估一次
# 6. 额外报告AUPRC（不参与模型选择）
# 7. 三个模型在相同5个seed下完整重复
# 8. 输出每次运行结果、mean±SD和按seed配对的性能差值

BATCH_SIZE = 8

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-2

POS_WEIGHT = 5.0

EPOCHS = 30

EARLY_STOPPING_PATIENCE = 10


# ============================================================
# 9. 通用预测与评价函数
# ============================================================

@torch.no_grad()
def collect_predictions(
        model,
        loader,
        desc
):

    model.eval()

    y_prob = []

    y_true = []


    for data in tqdm(
        loader,
        desc=desc,
        leave=False
    ):

        data = data.to(
            DEVICE
        )


        logits = model(
            data
        )


        probs = torch.sigmoid(
            logits
        )


        y_prob.append(
            probs
            .detach()
            .cpu()
            .numpy()
        )


        y_true.append(
            data.y
            .detach()
            .cpu()
            .numpy()
        )


    y_prob = np.concatenate(
        y_prob
    )


    y_true = np.concatenate(
        y_true
    )


    return y_true, y_prob


def calculate_ranking_metrics(
        y_true,
        y_prob
):

    return {

        "AUROC":
            roc_auc_score(
                y_true,
                y_prob
            ),

        "AUPRC":
            average_precision_score(
                y_true,
                y_prob
            )
    }


def select_best_f1_threshold(
        y_true,
        y_prob
):

    """
    只在validation set上选择threshold。

    sklearn precision_recall_curve返回：
        precision, recall: len(thresholds) + 1
        thresholds: len(thresholds)

    因此F1只使用precision[:-1]和recall[:-1]。
    """

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            y_prob
        )
    )


    if len(thresholds) == 0:

        return 0.5, 0.0


    precision_for_threshold = (
        precision[:-1]
    )

    recall_for_threshold = (
        recall[:-1]
    )


    denominator = (
        precision_for_threshold
        +
        recall_for_threshold
    )


    f1_values = np.divide(

        2.0
        *
        precision_for_threshold
        *
        recall_for_threshold,

        denominator,

        out=np.zeros_like(
            denominator,
            dtype=float
        ),

        where=(
            denominator > 0
        )
    )


    best_idx = int(
        np.argmax(
            f1_values
        )
    )


    best_threshold = float(
        thresholds[
            best_idx
        ]
    )


    best_val_f1 = float(
        f1_values[
            best_idx
        ]
    )


    return (
        best_threshold,
        best_val_f1
    )


def calculate_threshold_metrics(
        y_true,
        y_prob,
        threshold
):

    y_pred = (
        y_prob
        >=
        threshold
    ).astype(int)


    metrics = calculate_ranking_metrics(
        y_true,
        y_prob
    )


    metrics.update({

        "F1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "Precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "Accuracy":
            accuracy_score(
                y_true,
                y_pred
            )
    })


    return metrics


def safe_model_name(
        model_name
):

    safe = (
        model_name
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("\\", "_")
    )

    return safe


# ============================================================
# 10. 单个模型训练
# ============================================================

def train_and_eval(
        model_class,
        model_name,
        seed,
        use_edge_attr=False
):

    print("\n")
    print("=" * 70)

    print(
        "Training:",
        model_name,
        f"| Seed: {seed}"
    )

    print("=" * 70)


    # 每个模型在当前seed下重新固定随机状态。
    # 同一个seed会依次用于MLP、GCN和GATv2，形成公平的配对比较。
    seed_everything(
        seed
    )


    in_dim = (
        train_dataset[0]
        .x
        .shape[1]
    )


    if use_edge_attr:

        edge_dim = (
            train_dataset[0]
            .edge_attr
            .shape[1]
        )


        model = model_class(

            in_dim=in_dim,

            edge_dim=edge_dim

        ).to(
            DEVICE
        )


    else:

        model = model_class(

            in_dim=in_dim

        ).to(
            DEVICE
        )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    criterion = nn.BCEWithLogitsLoss(

        pos_weight=torch.tensor(
            [POS_WEIGHT],
            dtype=torch.float,
            device=DEVICE
        )
    )


    # 为当前seed单独创建DataLoader随机数生成器。
    # 这样同一个seed下三个模型不仅参数初始化可复现，
    # 训练样本shuffle顺序也保持一致，避免模型初始化消耗随机数
    # 导致不同模型看到不同的batch顺序。
    train_generator = torch.Generator()

    train_generator.manual_seed(
        seed
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        generator=train_generator
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0
    )


    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        f"{safe_model_name(model_name)}_seed_{seed}_best_val_auroc.pt"
    )


    best_val_auroc = -np.inf

    best_epoch = 0

    epochs_without_improvement = 0


    # ========================================================
    # Training + validation checkpoint selection
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0


        pbar = tqdm(

            train_loader,

            desc=(
                f"{model_name} "
                f"Epoch {epoch:02d}/{EPOCHS}"
            ),

            leave=False
        )


        for data in pbar:

            data = data.to(
                DEVICE
            )


            optimizer.zero_grad()


            logits = model(
                data
            )


            loss = criterion(

                logits,

                data.y.float()
            )


            loss.backward()


            optimizer.step()


            total_loss += (
                loss.item()
            )


        mean_loss = (
            total_loss
            /
            len(train_loader)
        )


        # ----------------------------------------------------
        # 每轮只在validation set上做模型选择
        # ----------------------------------------------------

        val_y_true, val_y_prob = (
            collect_predictions(
                model,
                val_loader,
                desc=f"Val epoch {epoch:02d}"
            )
        )


        val_ranking = calculate_ranking_metrics(
            val_y_true,
            val_y_prob
        )


        val_auroc = (
            val_ranking[
                "AUROC"
            ]
        )


        val_auprc = (
            val_ranking[
                "AUPRC"
            ]
        )


        print(
            f"{model_name:<26} "
            f"Epoch {epoch:02d} | "
            f"Train Loss={mean_loss:.5f} | "
            f"Val AUROC={val_auroc:.4f} | "
            f"Val AUPRC={val_auprc:.4f}"
        )


        # ----------------------------------------------------
        # 以Val AUROC作为唯一checkpoint选择标准
        # ----------------------------------------------------

        if val_auroc > best_val_auroc:

            best_val_auroc = float(
                val_auroc
            )

            best_epoch = epoch

            epochs_without_improvement = 0


            torch.save(
                model.state_dict(),
                checkpoint_path
            )


            print(
                f"  -> New best checkpoint: "
                f"epoch {best_epoch}, "
                f"Val AUROC={best_val_auroc:.4f}"
            )


        else:

            epochs_without_improvement += 1


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >=
            EARLY_STOPPING_PATIENCE
        ):

            print(
                f"  -> Early stopping at epoch {epoch}. "
                f"Best epoch={best_epoch}, "
                f"Best Val AUROC={best_val_auroc:.4f}"
            )

            break


    # ========================================================
    # 载入当前模型自己的最佳validation checkpoint
    # ========================================================

    try:

        best_state_dict = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True
        )

    except TypeError:

        best_state_dict = torch.load(
            checkpoint_path,
            map_location=DEVICE
        )


    model.load_state_dict(
        best_state_dict
    )


    # ========================================================
    # Validation：只用于threshold选择
    # ========================================================

    val_y_true, val_y_prob = (
        collect_predictions(
            model,
            val_loader,
            desc=f"{model_name} Best Val"
        )
    )


    best_threshold, best_val_f1 = (
        select_best_f1_threshold(
            val_y_true,
            val_y_prob
        )
    )


    val_metrics = calculate_threshold_metrics(
        val_y_true,
        val_y_prob,
        best_threshold
    )


    print(
        f"\n{model_name} Best Validation Checkpoint:"
    )

    print(
        f"Best Epoch      = "
        f"{best_epoch}"
    )

    print(
        f"Val AUROC       = "
        f"{val_metrics['AUROC']:.4f}"
    )

    print(
        f"Val AUPRC       = "
        f"{val_metrics['AUPRC']:.4f}"
    )

    print(
        f"Best Threshold  = "
        f"{best_threshold:.6f}"
    )

    print(
        f"Val F1@Threshold= "
        f"{best_val_f1:.4f}"
    )


    # ========================================================
    # Test：最佳checkpoint + validation threshold，只评估一次
    # ========================================================

    test_y_true, test_y_prob = (
        collect_predictions(
            model,
            test_loader,
            desc=f"{model_name} Test"
        )
    )


    test_metrics = calculate_threshold_metrics(
        test_y_true,
        test_y_prob,
        best_threshold
    )


    print(
        f"\n{model_name} Test:"
    )


    print(
        f"AUROC    = "
        f"{test_metrics['AUROC']:.4f}"
    )


    print(
        f"AUPRC    = "
        f"{test_metrics['AUPRC']:.4f}"
    )


    print(
        f"F1       = "
        f"{test_metrics['F1']:.4f}"
    )


    print(
        f"Precision= "
        f"{test_metrics['Precision']:.4f}"
    )


    print(
        f"Recall   = "
        f"{test_metrics['Recall']:.4f}"
    )


    print(
        f"Accuracy = "
        f"{test_metrics['Accuracy']:.4f}"
    )


    print(
        f"Threshold= "
        f"{best_threshold:.6f} "
        f"(selected on validation only)"
    )


    result = {

        "Seed":
            seed,

        "Best_Epoch":
            best_epoch,

        "Val_AUROC":
            val_metrics["AUROC"],

        "Val_AUPRC":
            val_metrics["AUPRC"],

        "Threshold":
            best_threshold,

        "AUROC":
            test_metrics["AUROC"],

        "AUPRC":
            test_metrics["AUPRC"],

        "F1":
            test_metrics["F1"],

        "Precision":
            test_metrics["Precision"],

        "Recall":
            test_metrics["Recall"],

        "Accuracy":
            test_metrics["Accuracy"],

        "Checkpoint":
            checkpoint_path
    }


    # 清GPU缓存
    del best_state_dict
    del model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return result


# ============================================================
# 11. 多seed汇总辅助函数
# ============================================================

def format_mean_sd(mean_value, sd_value, digits=4):

    return (
        f"{mean_value:.{digits}f} ± "
        f"{sd_value:.{digits}f}"
    )


def build_summary_dataframe(all_runs_df):

    """
    按Model汇总5个seed的均值和样本标准差（ddof=1）。
    """

    metrics = [
        "Best_Epoch",
        "Val_AUROC",
        "Val_AUPRC",
        "Threshold",
        "Test_AUROC",
        "Test_AUPRC",
        "Test_F1",
        "Test_Precision",
        "Test_Recall",
        "Test_Accuracy"
    ]

    summary_rows = []

    model_order = [
        "Feature-MLP",
        "Standard GCN",
        "Ours (Multi-scale GATv2)"
    ]

    for model_name in model_order:

        model_df = (
            all_runs_df[
                all_runs_df["Model"] == model_name
            ]
            .copy()
        )

        row = {
            "Model": model_name,
            "N_Seeds": len(model_df)
        }

        for metric in metrics:

            values = model_df[metric].astype(float)

            mean_value = float(
                values.mean()
            )

            sd_value = float(
                values.std(ddof=1)
            )

            row[f"{metric}_Mean"] = mean_value
            row[f"{metric}_SD"] = sd_value
            row[f"{metric}_Mean_SD"] = format_mean_sd(
                mean_value,
                sd_value,
                digits=(2 if metric == "Best_Epoch" else 4)
            )

        summary_rows.append(
            row
        )

    return pd.DataFrame(
        summary_rows
    )


def build_paired_delta_dataframe(all_runs_df):

    """
    在相同seed内计算Ours相对于两个baseline的配对差值。

    正值表示Ours更高。
    这里只做描述性统计，不进行显著性检验。
    """

    comparison_metrics = [
        "Test_AUROC",
        "Test_AUPRC",
        "Test_F1",
        "Test_Precision",
        "Test_Recall",
        "Test_Accuracy"
    ]

    ours_name = "Ours (Multi-scale GATv2)"

    baseline_names = [
        "Feature-MLP",
        "Standard GCN"
    ]

    delta_rows = []

    for seed in SEEDS:

        seed_df = (
            all_runs_df[
                all_runs_df["Seed"] == seed
            ]
            .set_index("Model")
        )

        if ours_name not in seed_df.index:

            raise RuntimeError(
                f"Seed {seed}: 缺少Ours结果"
            )

        for baseline_name in baseline_names:

            if baseline_name not in seed_df.index:

                raise RuntimeError(
                    f"Seed {seed}: 缺少{baseline_name}结果"
                )

            row = {
                "Seed": seed,
                "Comparison": f"Ours - {baseline_name}"
            }

            for metric in comparison_metrics:

                row[f"Delta_{metric}"] = float(
                    seed_df.loc[
                        ours_name,
                        metric
                    ]
                    -
                    seed_df.loc[
                        baseline_name,
                        metric
                    ]
                )

            delta_rows.append(
                row
            )

    return pd.DataFrame(
        delta_rows
    )


def build_paired_delta_summary(delta_df):

    delta_metrics = [
        c
        for c in delta_df.columns
        if c.startswith("Delta_")
    ]

    rows = []

    for comparison in delta_df["Comparison"].unique():

        sub = delta_df[
            delta_df["Comparison"] == comparison
        ]

        row = {
            "Comparison": comparison,
            "N_Seeds": len(sub)
        }

        for metric in delta_metrics:

            values = sub[metric].astype(float)

            mean_value = float(
                values.mean()
            )

            sd_value = float(
                values.std(ddof=1)
            )

            row[f"{metric}_Mean"] = mean_value
            row[f"{metric}_SD"] = sd_value
            row[f"{metric}_Mean_SD"] = format_mean_sd(
                mean_value,
                sd_value,
                digits=4
            )

            row[f"{metric}_Positive_Seeds"] = int(
                (values > 0).sum()
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 12. Main
# ============================================================

if __name__ == "__main__":

    all_rows = []

    model_specs = [
        (
            "Feature-MLP",
            Feature_MLP,
            False
        ),
        (
            "Standard GCN",
            Standard_GCN,
            False
        ),
        (
            "Ours (Multi-scale GATv2)",
            PPI_GAT_DualChain,
            True
        )
    ]


    # ========================================================
    # 5个seed逐一运行
    # ========================================================

    for seed_index, seed in enumerate(
        SEEDS,
        start=1
    ):

        print("\n")
        print("#" * 130)
        print(
            f"SEED {seed_index}/{len(SEEDS)}: {seed}"
        )
        print("#" * 130)


        for (
            model_name,
            model_class,
            use_edge_attr
        ) in model_specs:

            metrics = train_and_eval(
                model_class,
                model_name,
                seed=seed,
                use_edge_attr=use_edge_attr
            )

            all_rows.append({

                "Seed":
                    seed,

                "Model":
                    model_name,

                "Best_Epoch":
                    metrics["Best_Epoch"],

                "Val_AUROC":
                    metrics["Val_AUROC"],

                "Val_AUPRC":
                    metrics["Val_AUPRC"],

                "Threshold":
                    metrics["Threshold"],

                "Test_AUROC":
                    metrics["AUROC"],

                "Test_AUPRC":
                    metrics["AUPRC"],

                "Test_F1":
                    metrics["F1"],

                "Test_Precision":
                    metrics["Precision"],

                "Test_Recall":
                    metrics["Recall"],

                "Test_Accuracy":
                    metrics["Accuracy"],

                "Checkpoint":
                    metrics["Checkpoint"]
            })


            # 每完成一个模型就立刻写入临时结果，
            # 防止长时间5-seed运行过程中意外中断后完全丢失记录。
            pd.DataFrame(
                all_rows
            ).to_csv(
                os.path.join(
                    OUTPUT_DIR,
                    "internal_model_comparison_5seeds_all_runs.csv"
                ),
                index=False
            )


    # ========================================================
    # 全部单次运行结果
    # ========================================================

    all_runs_df = pd.DataFrame(
        all_rows
    )

    all_runs_path = os.path.join(
        OUTPUT_DIR,
        "internal_model_comparison_5seeds_all_runs.csv"
    )

    all_runs_df.to_csv(
        all_runs_path,
        index=False
    )


    # ========================================================
    # mean ± SD汇总
    # ========================================================

    summary_df = build_summary_dataframe(
        all_runs_df
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "internal_model_comparison_5seeds_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )


    # ========================================================
    # 按seed配对差值：Ours - baseline
    # ========================================================

    delta_df = build_paired_delta_dataframe(
        all_runs_df
    )

    delta_path = os.path.join(
        OUTPUT_DIR,
        "internal_model_comparison_5seeds_paired_deltas.csv"
    )

    delta_df.to_csv(
        delta_path,
        index=False
    )


    delta_summary_df = build_paired_delta_summary(
        delta_df
    )

    delta_summary_path = os.path.join(
        OUTPUT_DIR,
        "internal_model_comparison_5seeds_paired_delta_summary.csv"
    )

    delta_summary_df.to_csv(
        delta_summary_path,
        index=False
    )


    # ========================================================
    # 控制台：逐seed最终结果
    # ========================================================

    print("\n")
    print("=" * 145)
    print(
        "FINAL INTERNAL MODEL COMPARISON | "
        "5 RANDOM SEEDS | "
        "STRICT CLUSTER-DISJOINT TEST SET"
    )
    print("=" * 145)


    for seed in SEEDS:

        print(
            f"\nSeed = {seed}"
        )

        seed_df = all_runs_df[
            all_runs_df["Seed"] == seed
        ]

        for _, row in seed_df.iterrows():

            print(
                f"{row['Model']:<28} | "
                f"Best Epoch: {int(row['Best_Epoch']):>2d} | "
                f"Val AUROC: {row['Val_AUROC']:.4f} | "
                f"Threshold: {row['Threshold']:.4f} | "
                f"Test AUROC: {row['Test_AUROC']:.4f} | "
                f"AUPRC: {row['Test_AUPRC']:.4f} | "
                f"F1: {row['Test_F1']:.4f} | "
                f"Precision: {row['Test_Precision']:.4f} | "
                f"Recall: {row['Test_Recall']:.4f} | "
                f"Accuracy: {row['Test_Accuracy']:.4f}"
            )


    # ========================================================
    # 控制台：mean ± SD
    # ========================================================

    print("\n")
    print("=" * 145)
    print("MEAN ± SD ACROSS 5 RANDOM SEEDS")
    print("=" * 145)


    for _, row in summary_df.iterrows():

        print(
            f"{row['Model']:<28} | "
            f"Test AUROC: {row['Test_AUROC_Mean_SD']} | "
            f"AUPRC: {row['Test_AUPRC_Mean_SD']} | "
            f"F1: {row['Test_F1_Mean_SD']} | "
            f"Precision: {row['Test_Precision_Mean_SD']} | "
            f"Recall: {row['Test_Recall_Mean_SD']} | "
            f"Accuracy: {row['Test_Accuracy_Mean_SD']}"
        )


    # ========================================================
    # 控制台：配对差值
    # ========================================================

    print("\n")
    print("=" * 145)
    print("PAIRED DELTA ACROSS IDENTICAL SEEDS (OURS - BASELINE)")
    print("=" * 145)


    for _, row in delta_summary_df.iterrows():

        print(
            f"{row['Comparison']:<38} | "
            f"ΔAUROC: {row['Delta_Test_AUROC_Mean_SD']} "
            f"({int(row['Delta_Test_AUROC_Positive_Seeds'])}/{len(SEEDS)} seeds > 0) | "
            f"ΔAUPRC: {row['Delta_Test_AUPRC_Mean_SD']} "
            f"({int(row['Delta_Test_AUPRC_Positive_Seeds'])}/{len(SEEDS)} seeds > 0) | "
            f"ΔF1: {row['Delta_Test_F1_Mean_SD']} "
            f"({int(row['Delta_Test_F1_Positive_Seeds'])}/{len(SEEDS)} seeds > 0)"
        )


    print("=" * 145)

    print("\nSaved files:")
    print("1.", all_runs_path)
    print("2.", summary_path)
    print("3.", delta_path)
    print("4.", delta_summary_path)

    print("\nFinished.")
