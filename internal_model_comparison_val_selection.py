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

SEED = 42


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


seed_everything(SEED)


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

print("Internal model comparison")

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
# 本版本只修正模型选择与评价流程：
# 1. 每个epoch在validation set上计算AUROC
# 2. 依据Val AUROC保存各模型自己的最佳checkpoint
# 3. patience=10进行early stopping
# 4. 在最佳checkpoint的validation预测上选择F1最佳threshold
# 5. threshold固定后，test set只评估一次
# 6. 额外报告AUPRC（不参与模型选择）

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
        use_edge_attr=False
):

    print("\n")
    print("=" * 70)

    print(
        "Training:",
        model_name
    )

    print("=" * 70)


    # 每个模型重新固定seed，
    # 保证内部对比尽量公平
    seed_everything(
        SEED
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


    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0
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
        f"{safe_model_name(model_name)}_best_val_auroc.pt"
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
# 11. Main
# ============================================================

if __name__ == "__main__":

    results = {}


    # --------------------------------------------------------
    # Feature MLP
    # --------------------------------------------------------

    results[
        "Feature-MLP"
    ] = train_and_eval(

        Feature_MLP,

        "Feature-MLP",

        use_edge_attr=False
    )


    # --------------------------------------------------------
    # Standard GCN
    # --------------------------------------------------------

    results[
        "Standard GCN"
    ] = train_and_eval(

        Standard_GCN,

        "Standard GCN",

        use_edge_attr=False
    )


    # --------------------------------------------------------
    # Ours
    # --------------------------------------------------------

    results[
        "Ours (Multi-scale GATv2)"
    ] = train_and_eval(

        PPI_GAT_DualChain,

        "Ours (Multi-scale GATv2)",

        use_edge_attr=True
    )


    # ========================================================
    # Results
    # ========================================================

    print("\n")
    print("=" * 125)

    print(
        "Final internal model comparison "
        "(Strict Cluster-Disjoint Test Set | "
        "Best checkpoint selected by Validation AUROC)"
    )

    print("=" * 125)


    rows = []


    for model_name, metrics in results.items():

        print(

            f"{model_name:<28} | "

            f"Best Epoch: {metrics['Best_Epoch']:>2d} | "

            f"Val AUROC: {metrics['Val_AUROC']:.4f} | "

            f"Threshold: {metrics['Threshold']:.4f} | "

            f"Test AUROC: {metrics['AUROC']:.4f} | "

            f"AUPRC: {metrics['AUPRC']:.4f} | "

            f"F1: {metrics['F1']:.4f} | "

            f"Precision: {metrics['Precision']:.4f} | "

            f"Recall: {metrics['Recall']:.4f} | "

            f"Accuracy: {metrics['Accuracy']:.4f}"
        )


        rows.append({

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


    print("=" * 125)


    # ========================================================
    # 保存CSV
    # ========================================================

    result_df = pd.DataFrame(
        rows
    )


    result_path = os.path.join(

        OUTPUT_DIR,

        "internal_model_comparison_val_selected.csv"
    )


    result_df.to_csv(

        result_path,

        index=False

    )


    print(
        "\nResults saved to:"
    )

    print(
        result_path
    )
