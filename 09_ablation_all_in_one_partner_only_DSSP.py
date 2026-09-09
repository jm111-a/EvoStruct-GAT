# -*- coding: utf-8 -*-
r"""
EvoStruct-GAT: 9-group ablation experiments, all-in-one runner
==============================================================

Purpose
-------
Run all nine ablation experiments sequentially from the FINAL
partner-only DSSP 5 Å graphs, without modifying the formal graphs.

Final input
-----------
C:\Users\Administrator\Desktop\P-P
└─ cluster_split_v3_A2
   └─ partner_only_DSSP
      └─ 06_graphs
         ├─ train_pyg_3d_graph
         ├─ val_pyg_3d_graph
         └─ test_pyg_3d_graph

Output
------
C:\Users\Administrator\Desktop\P-P
└─ cluster_split_v3_A2
   └─ partner_only_DSSP
      └─ 09_ablation
         ├─ 01_without_distance_aware_edge_features
         ├─ 02_without_DSSP_derived_structural_features
         ├─ 03_without_dynamic_negative_node_sampling
         ├─ 04_without_ESM2_representations
         ├─ 05_without_focal_loss
         ├─ 06_without_focal_loss_and_undersampling
         ├─ 07_without_inter_residue_message_passing
         ├─ 08_without_multiscale_feature_fusion
         ├─ 09_without_residual_connections
         └─ ablation_summary.csv

Important
---------
1. The formal partner_only_DSSP/06_graphs are READ ONLY.
2. "Without DSSP" is implemented on-the-fly by setting x[:, 1280:1284] = 0
   after loading a graph. This is equivalent to generating a separate
   zero-DSSP graph copy, but avoids rewriting thousands of .pyg files.
3. All training settings that are not the target ablation are kept aligned
   with the formal 08train protocol:
       seed=42
       batch_size=4
       lr=1e-5
       epochs=100
       patience=20
       AdamW(weight_decay=1e-2)
       CosineAnnealingLR
       WeightedRandomSampler at graph level
       dynamic node negative sampling ratio 1:3 (unless ablated)
       FocalLoss(alpha=0.75, gamma=2.0) (unless ablated)
       best checkpoint selected by validation F1
       final threshold selected on validation set from 0.10 to 0.79
       test set evaluated using that fixed validation-selected threshold
"""

import os
import gc
import json
import random
import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, LayerNorm

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)

from tqdm import tqdm

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# ============================================================
# 0. User configuration
# ============================================================

ROOT = r"C:\Users\Administrator\Desktop\P-P"

GRAPH_ROOT = os.path.join(
    ROOT,
    "cluster_split_v3_A2",
    "partner_only_DSSP",
    "06_graphs",
)

ABLATION_ROOT = os.path.join(
    ROOT,
    "cluster_split_v3_A2",
    "partner_only_DSSP",
    "09_ablation",
)

os.makedirs(
    ABLATION_ROOT,
    exist_ok=True,
)


# Run all nine by default.
# If later you only want to rerun selected groups, for example:
# RUN_EXPERIMENT_NUMBERS = [2, 5, 9]
RUN_EXPERIMENT_NUMBERS = list(range(1, 10))


# If a group already has final_metrics.csv, skip it on rerun.
# This is useful if the machine is interrupted after several groups finish.
SKIP_COMPLETED = True


# Full input QC is recommended for the first run.
RUN_STARTUP_FULL_QC = True


# Save Train/Val learning curves for each ablation.
SAVE_TRAINING_PLOTS = True


# ============================================================
# 1. Formal training constants
# ============================================================

SEED = 42

BATCH_SIZE = 4
LEARNING_RATE = 1e-5
EPOCHS = 100
PATIENCE = 20

WEIGHT_DECAY = 1e-2

UNDERSAMPLE_RATIO = 3

FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

ESM_DIM = 1280
DSSP_DIM = 4
NODE_DIM = 1284
EDGE_DIM = 5

THRESHOLD_START = 0.10
THRESHOLD_STOP = 0.80
THRESHOLD_STEP = 0.01


EXPECTED_GRAPH_COUNTS = {
    "train": 2780,
    "val": 344,
    "test": 354,
}

EXPECTED_NODE_COUNTS = {
    "train": 620116,
    "val": 69673,
    "test": 74086,
}

EXPECTED_POSITIVE_COUNTS = {
    "train": 69700,
    "val": 10384,
    "test": 10437,
}


# ============================================================
# 2. Device and deterministic seed
# ============================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 3. Experiment definitions
# ============================================================

@dataclass(frozen=True)
class AblationConfig:

    number: int
    key: str
    folder: str
    display_name: str

    # Input / architecture ablations
    mask_dssp: bool = False
    zero_edge_attr: bool = False
    zero_esm2: bool = False
    disable_message_passing: bool = False
    disable_multiscale: bool = False
    disable_residual: bool = False

    # Training strategy ablations
    use_focal_loss: bool = True
    use_dynamic_negative_sampling: bool = True


EXPERIMENTS = [

    # 1
    AblationConfig(
        number=1,
        key="without_distance_aware_edge_features",
        folder="01_without_distance_aware_edge_features",
        display_name="Without Distance-Aware Edge Features",
        zero_edge_attr=True,
    ),

    # 2
    AblationConfig(
        number=2,
        key="without_DSSP_derived_structural_features",
        folder="02_without_DSSP_derived_structural_features",
        display_name="Without DSSP-derived Structural Features",
        mask_dssp=True,
    ),

    # 3
    AblationConfig(
        number=3,
        key="without_dynamic_negative_node_sampling",
        folder="03_without_dynamic_negative_node_sampling",
        display_name="Without Dynamic Negative-Node Sampling",
        use_focal_loss=True,
        use_dynamic_negative_sampling=False,
    ),

    # 4
    AblationConfig(
        number=4,
        key="without_ESM2_representations",
        folder="04_without_ESM2_representations",
        display_name="Without ESM-2 Representations",
        zero_esm2=True,
    ),

    # 5
    AblationConfig(
        number=5,
        key="without_focal_loss",
        folder="05_without_focal_loss",
        display_name="Without Focal Loss",
        use_focal_loss=False,
        use_dynamic_negative_sampling=True,
    ),

    # 6
    AblationConfig(
        number=6,
        key="without_focal_loss_and_undersampling",
        folder="06_without_focal_loss_and_undersampling",
        display_name="Without Focal Loss & Undersampling",
        use_focal_loss=False,
        use_dynamic_negative_sampling=False,
    ),

    # 7
    AblationConfig(
        number=7,
        key="without_inter_residue_message_passing",
        folder="07_without_inter_residue_message_passing",
        display_name="Without Inter-Residue Message Passing",
        disable_message_passing=True,
    ),

    # 8
    AblationConfig(
        number=8,
        key="without_multiscale_feature_fusion",
        folder="08_without_multiscale_feature_fusion",
        display_name="Without Multiscale Feature Fusion",
        disable_multiscale=True,
    ),

    # 9
    AblationConfig(
        number=9,
        key="without_residual_connections",
        folder="09_without_residual_connections",
        display_name="Without Residual Connections",
        disable_residual=True,
    ),
]


# ============================================================
# 4. Dataset
# ============================================================

class PPIGraphDataset(Dataset):

    def __init__(
            self,
            pyg_dir,
            mode="train",
            mask_dssp=False,
    ):

        super().__init__(
            None,
            None,
            None,
        )

        self.pyg_dir = pyg_dir
        self.mode = mode
        self.mask_dssp = mask_dssp

        if not os.path.isdir(
            pyg_dir
        ):

            raise FileNotFoundError(
                f"Graph directory does not exist:\n{pyg_dir}"
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
                f"No .pyg files found in {mode}:\n{pyg_dir}"
            )


    def len(self):

        return len(
            self.file_list
        )


    def get(
            self,
            idx,
    ):

        file_name = self.file_list[
            idx
        ]

        pyg_path = os.path.join(
            self.pyg_dir,
            file_name,
        )

        try:

            data = torch.load(
                pyg_path,
                weights_only=False,
            )

        except TypeError:

            data = torch.load(
                pyg_path,
            )

        # Basic checks
        if (
            not hasattr(data, "x")
            or
            data.x is None
        ):

            raise ValueError(
                f"{file_name}: missing x"
            )

        if (
            not hasattr(data, "y")
            or
            data.y is None
        ):

            raise ValueError(
                f"{file_name}: missing y"
            )

        if (
            not hasattr(data, "edge_index")
            or
            data.edge_index is None
        ):

            raise ValueError(
                f"{file_name}: missing edge_index"
            )

        if (
            not hasattr(data, "edge_attr")
            or
            data.edge_attr is None
        ):

            raise ValueError(
                f"{file_name}: missing edge_attr"
            )

        if data.x.ndim != 2:

            raise ValueError(
                f"{file_name}: abnormal x shape {data.x.shape}"
            )

        if data.x.shape[1] != NODE_DIM:

            raise ValueError(
                f"{file_name}: x dim = {data.x.shape[1]}, expected {NODE_DIM}"
            )

        if data.edge_attr.ndim != 2:

            raise ValueError(
                f"{file_name}: abnormal edge_attr shape {data.edge_attr.shape}"
            )

        if data.edge_attr.shape[1] != EDGE_DIM:

            raise ValueError(
                f"{file_name}: edge dim = {data.edge_attr.shape[1]}, expected {EDGE_DIM}"
            )

        if data.y.numel() != data.x.shape[0]:

            raise ValueError(
                f"{file_name}: node count != label count"
            )

        # --------------------------------------------------------
        # Ablation 2: w/o DSSP
        #
        # Old standalone code generated separate graphs with
        # x[:, 1280:1284] = 0. Here it is performed on-the-fly.
        # Formal source graphs remain untouched.
        # --------------------------------------------------------
        if self.mask_dssp:

            data = data.clone()

            data.x[
                :,
                ESM_DIM:
                ESM_DIM + DSSP_DIM
            ] = 0.0

        return data


# ============================================================
# 5. Dataset paths and base datasets
# ============================================================

TRAIN_DIR = os.path.join(
    GRAPH_ROOT,
    "train_pyg_3d_graph",
)

VAL_DIR = os.path.join(
    GRAPH_ROOT,
    "val_pyg_3d_graph",
)

TEST_DIR = os.path.join(
    GRAPH_ROOT,
    "test_pyg_3d_graph",
)


BASE_TRAIN_DATASET = PPIGraphDataset(
    TRAIN_DIR,
    mode="train",
    mask_dssp=False,
)

BASE_VAL_DATASET = PPIGraphDataset(
    VAL_DIR,
    mode="val",
    mask_dssp=False,
)

BASE_TEST_DATASET = PPIGraphDataset(
    TEST_DIR,
    mode="test",
    mask_dssp=False,
)


# ============================================================
# 6. Full dataset QC
# ============================================================

def dataset_qc(
        dataset,
        split_name,
):

    graph_count = len(
        dataset
    )

    node_count = 0
    positive_count = 0

    for i in tqdm(
        range(graph_count),
        desc=f"QC {split_name}",
    ):

        data = dataset[
            i
        ]

        node_count += int(
            data.y.numel()
        )

        positive_count += int(
            data.y.sum().item()
        )

    print(
        f"{split_name}: "
        f"graphs={graph_count}, "
        f"nodes={node_count}, "
        f"positive={positive_count}"
    )

    expected_graphs = EXPECTED_GRAPH_COUNTS[
        split_name
    ]

    expected_nodes = EXPECTED_NODE_COUNTS[
        split_name
    ]

    expected_positive = EXPECTED_POSITIVE_COUNTS[
        split_name
    ]

    if graph_count != expected_graphs:

        raise RuntimeError(
            f"{split_name}: graph count QC failed: "
            f"{graph_count} != {expected_graphs}"
        )

    if node_count != expected_nodes:

        raise RuntimeError(
            f"{split_name}: node count QC failed: "
            f"{node_count} != {expected_nodes}"
        )

    if positive_count != expected_positive:

        raise RuntimeError(
            f"{split_name}: positive count QC failed: "
            f"{positive_count} != {expected_positive}"
        )

    return {
        "split": split_name,
        "graphs": graph_count,
        "nodes": node_count,
        "positive": positive_count,
    }


# ============================================================
# 7. Graph-level WeightedRandomSampler weights
#
# This is part of the formal training protocol and is retained
# in ALL nine experiments, including experiment 6.
# "Undersampling" in experiment 6 refers to the dynamic
# node-level negative-node sampling, not graph-level sampling.
# ============================================================

def calculate_train_graph_weights(
        train_dataset,
):

    print(
        "\nCalculating graph-level interface-enhanced sampling weights..."
    )

    train_weights = []

    for i in tqdm(
        range(len(train_dataset)),
        desc="Graph weights",
    ):

        data = train_dataset[
            i
        ]

        pos_count = (
            data.y == 1
        ).sum().item()

        total_nodes = data.y.numel()

        weight = (
            pos_count + 1
        ) / total_nodes

        train_weights.append(
            weight
        )

    return train_weights


# ============================================================
# 8. Focal Loss
# ============================================================

class FocalLoss(nn.Module):

    def __init__(
            self,
            alpha=0.75,
            gamma=2.0,
    ):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma


    def forward(
            self,
            logits,
            targets,
    ):

        probs = torch.sigmoid(
            logits
        )

        ce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        p_t = (
            probs * targets
            +
            (1 - probs)
            *
            (1 - targets)
        )

        loss = (
            ce_loss
            *
            (
                (1 - p_t)
                **
                self.gamma
            )
        )

        if self.alpha >= 0:

            alpha_t = (
                self.alpha
                *
                targets
                +
                (1 - self.alpha)
                *
                (1 - targets)
            )

            loss = (
                alpha_t
                *
                loss
            )

        return loss.mean()


# ============================================================
# 9. Unified EvoStruct-GAT model
#
# The flags reproduce the architecture ablations supplied by
# the original standalone scripts.
# ============================================================

class PPI_GAT_Ablation(nn.Module):

    def __init__(
            self,
            in_dim=1284,
            edge_dim=5,
            esm_dim=1280,
            hidden_dim=256,
            heads=4,
            dropout=0.5,
            zero_edge_attr=False,
            zero_esm2=False,
            disable_message_passing=False,
            disable_multiscale=False,
            disable_residual=False,
    ):

        super().__init__()

        self.esm_dim = esm_dim

        self.struct_dim = (
            in_dim
            -
            esm_dim
        )

        self.dropout = dropout

        self.zero_edge_attr = zero_edge_attr
        self.zero_esm2 = zero_esm2
        self.disable_message_passing = disable_message_passing
        self.disable_multiscale = disable_multiscale
        self.disable_residual = disable_residual


        # ESM-2 branch
        self.esm_proj = nn.Sequential(

            nn.Linear(
                self.esm_dim,
                hidden_dim,
            ),

            nn.LayerNorm(
                hidden_dim,
            ),

            nn.SiLU(),

            nn.Dropout(
                dropout,
            ),
        )


        # DSSP branch
        self.struct_proj = nn.Sequential(

            nn.Linear(
                self.struct_dim,
                hidden_dim,
            ),

            nn.LayerNorm(
                hidden_dim,
            ),

            nn.SiLU(),

            nn.Dropout(
                dropout,
            ),
        )


        # Initial fusion
        self.initial_fuse = nn.Sequential(

            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),

            nn.LayerNorm(
                hidden_dim,
            ),

            nn.SiLU(),
        )


        # GATv2 layers
        self.gat1 = GATv2Conv(

            hidden_dim,

            hidden_dim // heads,

            heads=heads,

            edge_dim=edge_dim,

            dropout=dropout,
        )

        self.norm1 = LayerNorm(
            hidden_dim
        )


        self.gat2 = GATv2Conv(

            hidden_dim,

            hidden_dim // heads,

            heads=heads,

            edge_dim=edge_dim,

            dropout=dropout,
        )

        self.norm2 = LayerNorm(
            hidden_dim
        )


        # --------------------------------------------------------
        # Ablation 8:
        # w/o Multiscale Feature Fusion
        #
        # Old standalone version:
        # final_fuse input = hidden_dim and only h2 is used.
        # --------------------------------------------------------
        if self.disable_multiscale:

            self.final_fuse = nn.Sequential(

                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                ),

                nn.LayerNorm(
                    hidden_dim,
                ),

                nn.SiLU(),

                nn.Dropout(
                    dropout,
                ),
            )

        else:

            self.final_fuse = nn.Sequential(

                nn.Linear(
                    hidden_dim * 3,
                    hidden_dim,
                ),

                nn.LayerNorm(
                    hidden_dim,
                ),

                nn.SiLU(),

                nn.Dropout(
                    dropout,
                ),
            )


        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),

            nn.LayerNorm(
                hidden_dim // 2,
            ),

            nn.SiLU(),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(
                hidden_dim // 2,
                1,
            ),
        )


    def forward(
            self,
            data,
    ):

        x = data.x

        edge_index = (
            data.edge_index
        )

        edge_attr = (
            data.edge_attr
        )


        # --------------------------------------------------------
        # Ablation 1:
        # w/o distance-aware edge features.
        #
        # Preserve edge_index and edge_dim/parameters;
        # only zero edge_attr.
        # --------------------------------------------------------
        if self.zero_edge_attr:

            if edge_attr is not None:

                edge_attr = torch.zeros_like(
                    edge_attr
                )


        # Feature separation
        x_esm = x[
            :,
            :self.esm_dim
        ]

        x_struct = x[
            :,
            self.esm_dim:
        ]


        struct_emb = self.struct_proj(
            x_struct
        )


        # --------------------------------------------------------
        # Ablation 4:
        # w/o ESM-2 representations.
        #
        # Match the supplied standalone model:
        # zero the ESM embedding AFTER projection so Linear bias
        # cannot create a pseudo sequence signal.
        # --------------------------------------------------------
        if self.zero_esm2:

            esm_emb = torch.zeros_like(
                struct_emb
            )

        else:

            esm_emb = self.esm_proj(
                x_esm
            )


        # Initial fusion
        h_concat = torch.cat(
            [
                esm_emb,
                struct_emb,
            ],
            dim=-1,
        )

        h = self.initial_fuse(
            h_concat
        )


        # --------------------------------------------------------
        # Ablation 7:
        # w/o inter-residue message passing.
        #
        # Match supplied standalone model:
        # bypass both GATv2 layers and use
        # [esm_emb, h, zero_graph] as final multiscale input.
        # --------------------------------------------------------
        if self.disable_message_passing:

            zero_graph = torch.zeros_like(
                h
            )

            combined = torch.cat(
                [
                    esm_emb,
                    h,
                    zero_graph,
                ],
                dim=-1,
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


        # GAT layer 1
        h1 = self.gat1(
            h,
            edge_index,
            edge_attr,
        )

        h1 = self.norm1(
            h1
        )


        # --------------------------------------------------------
        # Ablation 9:
        # w/o residual connections.
        # --------------------------------------------------------
        if self.disable_residual:

            h1 = F.silu(
                h1
            )

        else:

            h1 = F.silu(
                h1 + h
            )


        # GAT layer 2
        h2 = self.gat2(
            h1,
            edge_index,
            edge_attr,
        )

        h2 = self.norm2(
            h2
        )


        if self.disable_residual:

            h2 = F.silu(
                h2
            )

        else:

            h2 = F.silu(
                h2 + h1
            )


        # --------------------------------------------------------
        # Ablation 8:
        # w/o multiscale feature fusion.
        # --------------------------------------------------------
        if self.disable_multiscale:

            out = self.final_fuse(
                h2
            )

        else:

            combined = torch.cat(
                [
                    esm_emb,
                    h1,
                    h2,
                ],
                dim=-1,
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
# 10. Model / parameter checks
# ============================================================

def count_parameters(
        model,
):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return (
        total,
        trainable,
    )


# ============================================================
# 11. DataLoaders for one experiment
# ============================================================

def build_datasets_for_experiment(
        config,
):

    if config.mask_dssp:

        train_dataset = PPIGraphDataset(
            TRAIN_DIR,
            mode="train",
            mask_dssp=True,
        )

        val_dataset = PPIGraphDataset(
            VAL_DIR,
            mode="val",
            mask_dssp=True,
        )

        test_dataset = PPIGraphDataset(
            TEST_DIR,
            mode="test",
            mask_dssp=True,
        )

    else:

        train_dataset = BASE_TRAIN_DATASET
        val_dataset = BASE_VAL_DATASET
        test_dataset = BASE_TEST_DATASET

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


def build_loaders(
        config,
        train_graph_weights,
):

    train_dataset, val_dataset, test_dataset = (
        build_datasets_for_experiment(
            config
        )
    )

    sampler = WeightedRandomSampler(
        train_graph_weights,
        num_samples=len(
            train_graph_weights
        ),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# 12. Metrics
# ============================================================

@torch.no_grad()
def get_metrics(
        model,
        loader,
        criterion,
        threshold=0.5,
):

    model.eval()

    all_probs = []
    all_labels = []

    total_loss = 0.0


    for data in loader:

        data = data.to(
            DEVICE
        )

        out = model(
            data
        )

        loss = criterion(
            out,
            data.y.float(),
        )

        total_loss += float(
            loss.item()
        )

        all_probs.append(
            torch.sigmoid(
                out
            )
            .detach()
            .cpu()
        )

        all_labels.append(
            data.y
            .detach()
            .cpu()
        )


    y_true = torch.cat(
        all_labels
    ).numpy()

    y_score = torch.cat(
        all_probs
    ).numpy()

    y_pred = (
        y_score > threshold
    ).astype(int)


    metrics = {

        "loss":
            total_loss
            /
            len(loader),

        "auroc":
            roc_auc_score(
                y_true,
                y_score,
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),
    }

    return (
        metrics,
        y_score,
        y_true,
    )


# ============================================================
# 13. Dynamic node negative sampling
# ============================================================

def compute_training_loss(
        config,
        criterion,
        logits,
        labels,
):

    # --------------------------------------------------------
    # Formal training:
    # all positive nodes +
    # up to 3x randomly sampled negative nodes.
    # --------------------------------------------------------
    if config.use_dynamic_negative_sampling:

        pos_mask = (
            labels == 1
        )

        neg_mask = (
            labels == 0
        )

        neg_indices = (
            neg_mask
            .nonzero(
                as_tuple=True
            )[0]
        )

        num_neg = min(
            len(
                neg_indices
            ),
            pos_mask.sum().item()
            *
            UNDERSAMPLE_RATIO,
        )

        if num_neg > 0:

            # Deliberately kept aligned with the formal 08train logic.
            perm = torch.randperm(
                len(
                    neg_indices
                )
            )[
                :num_neg
            ]

            selected_neg_indices = (
                neg_indices[
                    perm
                ]
            )

            final_mask = torch.zeros_like(
                labels,
                dtype=torch.bool,
            )

            final_mask[
                pos_mask
            ] = True

            final_mask[
                selected_neg_indices
            ] = True

            return criterion(
                logits[
                    final_mask
                ],
                labels[
                    final_mask
                ].float(),
            )

    # --------------------------------------------------------
    # Ablations 3 and 6:
    # all nodes participate in the loss.
    # --------------------------------------------------------
    return criterion(
        logits,
        labels.float(),
    )


# ============================================================
# 14. Validation threshold selection
# ============================================================

def find_best_validation_threshold(
        val_probs,
        val_labels,
):

    best_threshold = 0.5
    max_f1 = 0.0


    for t in np.arange(
        THRESHOLD_START,
        THRESHOLD_STOP,
        THRESHOLD_STEP,
    ):

        current_f1 = f1_score(
            val_labels,
            (
                val_probs > t
            ).astype(int),
            zero_division=0,
        )

        if current_f1 > max_f1:

            max_f1 = current_f1

            best_threshold = float(
                t
            )


    return (
        best_threshold,
        max_f1,
    )


# ============================================================
# 15. Final test metrics
# ============================================================

def calculate_final_test_metrics(
        test_probs,
        test_labels,
        threshold,
):

    y_pred_test = (
        test_probs > threshold
    ).astype(int)

    return {

        "threshold":
            float(
                threshold
            ),

        "AUROC":
            float(
                roc_auc_score(
                    test_labels,
                    test_probs,
                )
            ),

        "F1":
            float(
                f1_score(
                    test_labels,
                    y_pred_test,
                    zero_division=0,
                )
            ),

        "Precision":
            float(
                precision_score(
                    test_labels,
                    y_pred_test,
                    zero_division=0,
                )
            ),

        "Recall":
            float(
                recall_score(
                    test_labels,
                    y_pred_test,
                    zero_division=0,
                )
            ),

        "Accuracy":
            float(
                accuracy_score(
                    test_labels,
                    y_pred_test,
                )
            ),
    }


# ============================================================
# 16. Training history
# ============================================================

def flatten_history(
        history,
):

    rows = []

    for e in history:

        row = {
            "epoch":
                e["epoch"],
        }

        for split in [
            "train",
            "val",
            "test",
        ]:

            for metric_name, value in e[
                split
            ].items():

                row[
                    f"{split}_{metric_name}"
                ] = value

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 17. Training curves
#
# Uses matplotlib defaults; one chart per figure.
# ============================================================

def save_training_plots(
        history_df,
        output_dir,
        best_epoch,
):

    if not SAVE_TRAINING_PLOTS:

        return

    plot_specs = [

        (
            "loss",
            "Loss",
            "training_curve_loss.png",
        ),

        (
            "auroc",
            "AUROC",
            "training_curve_auroc.png",
        ),

        (
            "f1",
            "F1-Score",
            "training_curve_f1.png",
        ),
    ]


    for metric, ylabel, file_name in plot_specs:

        fig = plt.figure(
            figsize=(7.0, 5.0)
        )

        ax = fig.add_subplot(
            111
        )

        ax.plot(
            history_df["epoch"],
            history_df[
                f"train_{metric}"
            ],
            label="Training",
        )

        ax.plot(
            history_df["epoch"],
            history_df[
                f"val_{metric}"
            ],
            label="Validation",
        )

        ax.axvline(
            best_epoch,
            linestyle="--",
            label=f"Best epoch ({best_epoch})",
        )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                output_dir,
                file_name,
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )


# ============================================================
# 18. Save experiment design
# ============================================================

def save_experiment_design():

    rows = []

    for config in EXPERIMENTS:

        row = asdict(
            config
        )

        row.update({

            "graph_level_weighted_sampler":
                True,

            "dynamic_negative_ratio":
                (
                    UNDERSAMPLE_RATIO
                    if
                    config.use_dynamic_negative_sampling
                    else
                    "all_nodes"
                ),

            "loss":
                (
                    "FocalLoss(alpha=0.75,gamma=2.0)"
                    if
                    config.use_focal_loss
                    else
                    "BCEWithLogitsLoss"
                ),

            "seed":
                SEED,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "epochs":
                EPOCHS,

            "patience":
                PATIENCE,
        })

        rows.append(
            row
        )

    pd.DataFrame(
        rows
    ).to_csv(
        os.path.join(
            ABLATION_ROOT,
            "ablation_design.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 19. Train one ablation experiment
# ============================================================

def run_one_experiment(
        config,
        train_graph_weights,
):

    output_dir = os.path.join(
        ABLATION_ROOT,
        config.folder,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    final_metrics_path = os.path.join(
        output_dir,
        "final_metrics.csv",
    )


    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------
    if (
        SKIP_COMPLETED
        and
        os.path.exists(
            final_metrics_path
        )
    ):

        print(
            "\n"
            +
            "=" * 78
        )

        print(
            f"SKIP completed: "
            f"{config.number}. "
            f"{config.display_name}"
        )

        print(
            final_metrics_path
        )

        print(
            "=" * 78
        )

        result_df = pd.read_csv(
            final_metrics_path
        )

        if len(
            result_df
        ) != 1:

            raise RuntimeError(
                f"Unexpected final_metrics.csv: "
                f"{final_metrics_path}"
            )

        return result_df.iloc[
            0
        ].to_dict()


    # --------------------------------------------------------
    # Critical:
    # reset seed BEFORE each group so the nine standalone
    # experiments start from the same deterministic seed.
    # --------------------------------------------------------
    seed_everything(
        SEED
    )


    print(
        "\n\n"
        +
        "#" * 84
    )

    print(
        f"ABLATION {config.number}/9"
    )

    print(
        config.display_name
    )

    print(
        "#" * 84
    )

    print(
        "Input graph root:"
    )

    print(
        GRAPH_ROOT
    )

    print(
        "Output:"
    )

    print(
        output_dir
    )

    print(
        "\nConfiguration:"
    )

    print(
        json.dumps(
            asdict(config),
            indent=2,
            ensure_ascii=False,
        )
    )


    # --------------------------------------------------------
    # Datasets / loaders
    # --------------------------------------------------------
    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_loaders(
        config,
        train_graph_weights,
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    sample = train_dataset[
        0
    ]

    model = PPI_GAT_Ablation(

        in_dim=sample.x.shape[
            1
        ],

        edge_dim=sample.edge_attr.shape[
            1
        ],

        zero_edge_attr=
            config.zero_edge_attr,

        zero_esm2=
            config.zero_esm2,

        disable_message_passing=
            config.disable_message_passing,

        disable_multiscale=
            config.disable_multiscale,

        disable_residual=
            config.disable_residual,

    ).to(
        DEVICE
    )


    total_params, trainable_params = (
        count_parameters(
            model
        )
    )


    print(
        f"\nModel parameters: "
        f"total={total_params:,}, "
        f"trainable={trainable_params:,}"
    )


    # --------------------------------------------------------
    # Optimizer / scheduler / criterion
    # --------------------------------------------------------
    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,
    )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=EPOCHS,
    )


    if config.use_focal_loss:

        criterion = FocalLoss(

            alpha=FOCAL_ALPHA,

            gamma=FOCAL_GAMMA,
        )

        criterion_name = (
            f"FocalLoss(alpha={FOCAL_ALPHA},"
            f" gamma={FOCAL_GAMMA})"
        )

    else:

        criterion = nn.BCEWithLogitsLoss()

        criterion_name = (
            "BCEWithLogitsLoss"
        )


    print(
        "Loss:",
        criterion_name
    )

    print(
        "Dynamic negative-node sampling:",
        config.use_dynamic_negative_sampling
    )

    print(
        "Graph-level WeightedRandomSampler:",
        True
    )


    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------
    best_model_path = os.path.join(
        output_dir,
        "best_model.pth",
    )

    history_path = os.path.join(
        output_dir,
        "training_history.csv",
    )

    config_path = os.path.join(
        output_dir,
        "experiment_config.json",
    )


    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                **asdict(config),
                "seed":
                    SEED,
                "batch_size":
                    BATCH_SIZE,
                "learning_rate":
                    LEARNING_RATE,
                "weight_decay":
                    WEIGHT_DECAY,
                "epochs":
                    EPOCHS,
                "patience":
                    PATIENCE,
                "undersample_ratio":
                    UNDERSAMPLE_RATIO,
                "criterion":
                    criterion_name,
                "graph_root":
                    GRAPH_ROOT,
                "total_parameters":
                    total_params,
                "trainable_parameters":
                    trainable_params,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    best_val_f1 = 0.0
    best_epoch = 0

    patience_counter = 0

    history = []


    print(
        "\nStarting training..."
    )


    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        train_loss_running = 0.0


        pbar = tqdm(

            train_loader,

            desc=(
                f"[{config.number}/9] "
                f"Epoch {epoch:03d}/{EPOCHS}"
            ),
        )


        for data in pbar:

            data = data.to(
                DEVICE
            )

            optimizer.zero_grad()

            out = model(
                data
            )

            loss = compute_training_loss(

                config,

                criterion,

                out,

                data.y,
            )

            loss.backward()

            optimizer.step()

            train_loss_running += float(
                loss.item()
            )


        scheduler.step()


        # ----------------------------------------------------
        # Keep formal 08train evaluation structure:
        # train / val / test are logged every epoch,
        # but ONLY validation F1 controls checkpointing.
        # ----------------------------------------------------
        train_m, _, _ = get_metrics(

            model,

            train_loader,

            criterion,
        )


        val_m, _, _ = get_metrics(

            model,

            val_loader,

            criterion,
        )


        test_m, _, _ = get_metrics(

            model,

            test_loader,

            criterion,
        )


        history.append({

            "epoch":
                epoch,

            "train":
                train_m,

            "val":
                val_m,

            "test":
                test_m,
        })


        # Save history after every epoch for crash recovery.
        flatten_history(
            history
        ).to_csv(
            history_path,
            index=False,
            encoding="utf-8-sig",
        )


        mean_training_step_loss = (
            train_loss_running
            /
            len(train_loader)
        )


        print(

            f"Ep {epoch:03d} | "

            f"step_train_loss="
            f"{mean_training_step_loss:.4f} | "

            f"Val F1="
            f"{val_m['f1']:.4f} | "

            f"Val Loss="
            f"{val_m['loss']:.4f} | "

            f"Val AUROC="
            f"{val_m['auroc']:.4f}"
        )


        # ----------------------------------------------------
        # Best model selected ONLY by validation F1
        # ----------------------------------------------------
        if val_m[
            "f1"
        ] > best_val_f1:

            best_val_f1 = float(
                val_m[
                    "f1"
                ]
            )

            best_epoch = epoch

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            patience_counter = 0

            print(
                f"Best model updated: "
                f"epoch={best_epoch}, "
                f"val_F1={best_val_f1:.4f}"
            )


        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    f"Early stopping: "
                    f"{PATIENCE} epochs without "
                    f"validation F1 improvement."
                )

                break


    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------
    if not os.path.exists(
        best_model_path
    ):

        raise RuntimeError(
            f"Best model was not saved:\n"
            f"{best_model_path}"
        )


    try:

        state_dict = torch.load(

            best_model_path,

            map_location=DEVICE,

            weights_only=True,
        )

    except TypeError:

        state_dict = torch.load(

            best_model_path,

            map_location=DEVICE,
        )


    model.load_state_dict(
        state_dict
    )


    # --------------------------------------------------------
    # Validation threshold search and final test
    # --------------------------------------------------------
    _, val_probs, val_labels = get_metrics(

        model,

        val_loader,

        criterion,
    )


    _, test_probs, test_labels = get_metrics(

        model,

        test_loader,

        criterion,
    )


    best_threshold, best_threshold_val_f1 = (
        find_best_validation_threshold(

            val_probs,

            val_labels,
        )
    )


    final_test = calculate_final_test_metrics(

        test_probs,

        test_labels,

        best_threshold,
    )


    # --------------------------------------------------------
    # Final record
    # --------------------------------------------------------
    final_record = {

        "Order":
            config.number,

        "Ablation":
            config.display_name,

        "Key":
            config.key,

        "Best_Epoch":
            best_epoch,

        "Best_Val_F1_at_0.5":
            best_val_f1,

        "Validation_Selected_Threshold":
            final_test[
                "threshold"
            ],

        "Validation_F1_at_Selected_Threshold":
            best_threshold_val_f1,

        "AUROC":
            final_test[
                "AUROC"
            ],

        "F1":
            final_test[
                "F1"
            ],

        "Precision":
            final_test[
                "Precision"
            ],

        "Recall":
            final_test[
                "Recall"
            ],

        "Accuracy":
            final_test[
                "Accuracy"
            ],

        "Total_Parameters":
            total_params,

        "Trainable_Parameters":
            trainable_params,

        "Focal_Loss":
            config.use_focal_loss,

        "Dynamic_Negative_Node_Sampling":
            config.use_dynamic_negative_sampling,

        "DSSP_Masked":
            config.mask_dssp,

        "Edge_Attributes_Zeroed":
            config.zero_edge_attr,

        "ESM2_Zeroed":
            config.zero_esm2,

        "Message_Passing_Disabled":
            config.disable_message_passing,

        "Multiscale_Fusion_Disabled":
            config.disable_multiscale,

        "Residual_Connections_Disabled":
            config.disable_residual,
    }


    pd.DataFrame(
        [
            final_record
        ]
    ).to_csv(
        final_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )


    with open(
        os.path.join(
            output_dir,
            "final_metrics.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_record,
            f,
            indent=2,
            ensure_ascii=False,
        )


    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    history_df = flatten_history(
        history
    )

    save_training_plots(

        history_df,

        output_dir,

        best_epoch,
    )


    # --------------------------------------------------------
    # Console final metrics
    # --------------------------------------------------------
    print(
        "\n"
        +
        "=" * 78
    )

    print(
        f"FINAL | {config.display_name}"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Validation-selected threshold: "
        f"{final_test['threshold']:.3f}"
    )

    print(
        f"AUROC:    {final_test['AUROC']:.4f}"
    )

    print(
        f"F1:       {final_test['F1']:.4f}"
    )

    print(
        f"Precision:{final_test['Precision']:.4f}"
    )

    print(
        f"Recall:   {final_test['Recall']:.4f}"
    )

    print(
        f"Accuracy: {final_test['Accuracy']:.4f}"
    )

    print(
        "=" * 78
    )


    # --------------------------------------------------------
    # Cleanup before next ablation
    # --------------------------------------------------------
    del model
    del optimizer
    del scheduler
    del criterion

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return final_record


# ============================================================
# 20. Main
# ============================================================

def main():

    print(
        "=" * 88
    )

    print(
        "EvoStruct-GAT | 9 Ablation Experiments | All-in-One"
    )

    print(
        "=" * 88
    )

    print(
        "Device:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    print(
        "\nFormal graph root:"
    )

    print(
        GRAPH_ROOT
    )

    print(
        "\nAblation output root:"
    )

    print(
        ABLATION_ROOT
    )

    print(
        "\nImportant: formal graphs are never overwritten."
    )


    # --------------------------------------------------------
    # Save experiment design
    # --------------------------------------------------------
    save_experiment_design()


    # --------------------------------------------------------
    # Startup full QC
    # --------------------------------------------------------
    if RUN_STARTUP_FULL_QC:

        print(
            "\n"
            +
            "=" * 88
        )

        print(
            "STARTUP FULL QC"
        )

        print(
            "=" * 88
        )

        qc_rows = []

        qc_rows.append(
            dataset_qc(
                BASE_TRAIN_DATASET,
                "train",
            )
        )

        qc_rows.append(
            dataset_qc(
                BASE_VAL_DATASET,
                "val",
            )
        )

        qc_rows.append(
            dataset_qc(
                BASE_TEST_DATASET,
                "test",
            )
        )

        pd.DataFrame(
            qc_rows
        ).to_csv(
            os.path.join(
                ABLATION_ROOT,
                "formal_5A_graph_QC.csv",
            ),
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "\nFormal partner-only 5A graph QC: PASS"
        )


    # --------------------------------------------------------
    # Calculate formal graph-level sampling weights once.
    # Labels are identical across all 9 ablations.
    # --------------------------------------------------------
    seed_everything(
        SEED
    )

    train_graph_weights = (
        calculate_train_graph_weights(
            BASE_TRAIN_DATASET
        )
    )


    # --------------------------------------------------------
    # Run selected experiments in the exact requested order
    # --------------------------------------------------------
    selected_experiments = [

        config

        for config in EXPERIMENTS

        if config.number
        in
        RUN_EXPERIMENT_NUMBERS
    ]


    if len(
        selected_experiments
    ) == 0:

        raise ValueError(
            "RUN_EXPERIMENT_NUMBERS is empty or invalid."
        )


    summary_records = []


    for config in selected_experiments:

        result = run_one_experiment(

            config,

            train_graph_weights,
        )

        summary_records.append(
            result
        )


        # Update summary after each completed experiment.
        summary_df = pd.DataFrame(
            summary_records
        )

        if "Order" in summary_df.columns:

            summary_df = (
                summary_df
                .sort_values(
                    "Order"
                )
                .reset_index(
                    drop=True
                )
            )

        summary_df.to_csv(
            os.path.join(
                ABLATION_ROOT,
                "ablation_summary.csv",
            ),
            index=False,
            encoding="utf-8-sig",
        )


    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------
    summary_df = pd.DataFrame(
        summary_records
    )

    if "Order" in summary_df.columns:

        summary_df = (
            summary_df
            .sort_values(
                "Order"
            )
            .reset_index(
                drop=True
            )
        )


    final_summary_path = os.path.join(
        ABLATION_ROOT,
        "ablation_summary.csv",
    )


    summary_df.to_csv(
        final_summary_path,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        "\n\n"
        +
        "=" * 110
    )

    print(
        "ALL REQUESTED ABLATION EXPERIMENTS FINISHED"
    )

    print(
        "=" * 110
    )


    display_columns = [

        "Order",

        "Ablation",

        "Validation_Selected_Threshold",

        "AUROC",

        "F1",

        "Precision",

        "Recall",

        "Accuracy",
    ]


    existing_display_columns = [

        c

        for c in display_columns

        if c in summary_df.columns
    ]


    print(
        summary_df[
            existing_display_columns
        ].to_string(
            index=False
        )
    )


    print(
        "\nSummary saved to:"
    )

    print(
        final_summary_path
    )

    print(
        "=" * 110
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()
