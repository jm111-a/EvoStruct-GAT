import os
import re
import warnings
import ctypes
import importlib

import torch
import esm
import numpy as np
import pandas as pd

from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.DSSP import DSSP
from Bio.SeqUtils import seq1
from torch_geometric.data import Data

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# 0. Windows / warnings
# ============================================================

if os.name == "nt":
    ctypes.windll.kernel32.SetErrorMode(0x0002 | 0x8000)

warnings.filterwarnings("ignore")


# ============================================================
# 1. 基础配置
# ============================================================

ROOT = r"C:\Users\Administrator\Desktop\P-P"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# corrected 5A 正式模型权重
WEIGHTS_PATH = r"C:\Users\Administrator\Desktop\P-P\cluster_split_v3_A2\partner_only_DSSP\07_model_save\best_model.pth"

# 当前正式模型架构文件：07_model.py
MODEL_MODULE_NAME = "07_model"

# DSSP
DSSP_EXE_PATH = os.path.join(
    ROOT,
    "dssp.exe",
)

# CR1 Top-K case study 输出目录
# 单独保存，避免覆盖之前的threshold分析结果
OUTPUT_DIR = r"C:\Users\Administrator\Desktop\P-P\cluster_split_v3_A2\partner_only_DSSP\11_CR1_case_analysis\CR1_case_analysis_old_model_topK"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

# 与正式 graph 构建保持一致
GRAPH_CUTOFF = 10.0
ESM_MAX_LEN = 1022
TOTAL_NODE_DIM = 1284
EDGE_DIM = 5

# Top-K ranking analysis
# Figure 7 原始逻辑本质上是按预测概率排序后选取高排名候选，
# 因此本脚本不再使用分类阈值筛选残基，而是保留全部扫描残基并按概率降序排序。
TOP_K_VALUES = [5, 8, 10, 20, 30]

# 自动从参考复合物生成 ground truth 时，采用正式 5A 定义：
# target residue 与 partner chain(s) 任一非氢原子距离 < 5.0 Å
GROUND_TRUTH_CONTACT_CUTOFF = 5.0


# ============================================================
# 2. 加载正式模型架构
# ============================================================

print("=" * 72)
print("Old formal 5A model + CR1 probability ranking + Top-K evaluation")
print("=" * 72)
print("Device:", DEVICE)
print("Weights:", WEIGHTS_PATH)
print("Model module:", MODEL_MODULE_NAME)
print("Output:", OUTPUT_DIR)

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"找不到 corrected 5A 正式模型权重：\n{WEIGHTS_PATH}"
    )

try:
    model_module = importlib.import_module(MODEL_MODULE_NAME)

    if hasattr(model_module, "PPI_GAT_DualChain"):
        PPI_Model = getattr(model_module, "PPI_GAT_DualChain")
    elif hasattr(model_module, "PPI_GAT_MultiModal"):
        PPI_Model = getattr(model_module, "PPI_GAT_MultiModal")
    else:
        raise ImportError(
            f"未在 {MODEL_MODULE_NAME}.py 中找到 "
            f"PPI_GAT_DualChain 或 PPI_GAT_MultiModal"
        )

except Exception as e:
    raise RuntimeError(f"模型模块加载失败：{e}")


# ============================================================
# 3. 通用工具
# ============================================================

STANDARD_AA = {
    "ALA", "CYS", "ASP", "GLU", "PHE",
    "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG",
    "SER", "THR", "VAL", "TRP", "TYR",
}


def parse_chain_ids(chain_string):
    """
    支持：
        A
        HL
        A,B
        A B
        A;B
    """
    s = str(chain_string)
    s = (
        s
        .replace(",", "")
        .replace(" ", "")
        .replace(";", "")
        .strip()
    )
    return list(s)


def residue_uid(residue):
    """
    将 PDB residue id 转成稳定字符串。

    示例：
        140  -> "140"
        140A -> "140A"
    """
    seq_num = residue.get_id()[1]
    insertion_code = str(residue.get_id()[2]).strip()

    if insertion_code:
        return f"{seq_num}{insertion_code}"

    return str(seq_num)


def is_standard_residue(residue):
    return (
        residue.get_id()[0] == " "
        and residue.get_resname() in STANDARD_AA
    )


def is_heavy_atom(atom):
    """排除 H / D。"""
    element = str(getattr(atom, "element", "")).strip().upper()
    atom_name = str(atom.get_name()).strip().upper()

    if element in {"H", "D"}:
        return False

    # 某些旧 PDB 的 element 字段可能不规范，再用 atom name 做保护。
    if atom_name.startswith("H"):
        return False

    return True


def parse_manual_ground_truth(text):
    """
    支持：
        91,140,141,1114
        91 140 141 1114
        140A,141
    """
    tokens = re.split(r"[,;\s]+", text.strip())

    return {
        token.strip()
        for token in tokens
        if token.strip()
    }


# ============================================================
# 4. 构建预测 graph
#
# 与正式训练保持一致：
# ESM2 1280 + DSSP 4 = 1284
# C-alpha < 10 Å
# Gaussian edge features = 5 dims
# ============================================================

def build_graph_for_prediction(
        pdb_path,
        target_chain,
        model_esm,
        batch_converter):

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure(
        "target",
        pdb_path,
    )

    model_struct = structure[0]

    if target_chain not in model_struct:
        raise ValueError(
            f"PDB 中未找到链 '{target_chain}'"
        )

    all_residues = []

    for res in model_struct[target_chain]:
        if is_standard_residue(res):
            all_residues.append(res)

    if not all_residues:
        raise ValueError(
            f"链 '{target_chain}' 中没有有效标准氨基酸残基"
        )

    # --------------------------------------------------------
    # DSSP
    # --------------------------------------------------------

    dssp_dict = None

    try:
        dssp_obj = DSSP(
            model_struct,
            pdb_path,
            dssp=DSSP_EXE_PATH,
        )
        dssp_dict = dict(dssp_obj)

    except Exception as e:
        print(
            f"WARNING: DSSP 计算失败，将使用默认 Coil/RSA 特征兜底：{e}"
        )

    # --------------------------------------------------------
    # Sequence
    # --------------------------------------------------------

    seq = ""

    for res in all_residues:
        try:
            aa = seq1(
                res.get_resname(),
                custom_map={"MSE": "M"},
            )
        except Exception:
            aa = "X"

        if aa in {"", "?"}:
            aa = "X"

        seq += aa

    # --------------------------------------------------------
    # ESM2 最大长度 1022
    # --------------------------------------------------------

    if len(seq) > ESM_MAX_LEN:
        print(
            f"WARNING: 链 {target_chain} 长度 {len(seq)} > {ESM_MAX_LEN}，"
            f"超出部分将按照正式流程截断。"
        )
        seq = seq[:ESM_MAX_LEN]

    data_esm = [("target", seq)]

    _, _, batch_tokens = batch_converter(data_esm)
    batch_tokens = batch_tokens.to(DEVICE)

    with torch.no_grad():
        results = model_esm(
            batch_tokens,
            repr_layers=[33],
            return_contacts=False,
        )

        token_representations = (
            results["representations"][33][
                0,
                1:len(seq) + 1,
            ]
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Node features
    # --------------------------------------------------------

    node_features = []
    node_coords = []
    res_mapping_info = []

    valid_count = min(
        len(all_residues),
        len(token_representations),
    )

    for i in range(valid_count):
        res = all_residues[i]

        chain_id = res.get_full_id()[2]
        res_num = res.get_id()[1]
        insertion_code = str(res.get_id()[2]).strip()
        res_name = res.get_resname()
        uid = residue_uid(res)

        # ----------------------------------------------------
        # DSSP 4维：[Helix, Sheet, Coil, RSA]
        #
        # 与正式训练定义一致：
        # H/G/I -> Helix
        # B/E   -> Sheet
        # 其余   -> Coil
        # ----------------------------------------------------

        struct_feat = [0.0, 0.0, 1.0, 0.5]

        if dssp_dict is not None:
            dssp_key = (chain_id, res.get_id())

            if dssp_key in dssp_dict:
                dssp_val = dssp_dict[dssp_key]
                sec_struct_char = dssp_val[2]
                rsa = dssp_val[3]

                if sec_struct_char in ["H", "G", "I"]:
                    ss_feat = [1.0, 0.0, 0.0]
                elif sec_struct_char in ["B", "E"]:
                    ss_feat = [0.0, 1.0, 0.0]
                else:
                    ss_feat = [0.0, 0.0, 1.0]

                try:
                    rsa = float(rsa)
                except (ValueError, TypeError):
                    rsa = 0.5

                struct_feat = [
                    ss_feat[0],
                    ss_feat[1],
                    ss_feat[2],
                    rsa,
                ]

        full_feat = np.concatenate(
            [
                token_representations[i],
                np.asarray(struct_feat, dtype=np.float32),
            ]
        )

        if "CA" in res:
            coord = res["CA"].get_coord()
        else:
            coord = res.child_list[0].get_coord()

        node_features.append(full_feat)
        node_coords.append(coord)

        res_mapping_info.append(
            {
                "chain": chain_id,
                "res_name": res_name,
                "pdb_id": res_num,
                "insertion_code": insertion_code,
                "residue_uid": uid,
                "coord": np.asarray(coord, dtype=float),
            }
        )

    x = torch.tensor(
        np.asarray(node_features),
        dtype=torch.float,
    )

    pos = torch.tensor(
        np.asarray(node_coords),
        dtype=torch.float,
    )

    if x.ndim != 2 or x.shape[1] != TOTAL_NODE_DIM:
        raise ValueError(
            f"x feature dimension 异常：{x.shape}，expected (*, {TOTAL_NODE_DIM})"
        )

    # --------------------------------------------------------
    # C-alpha < 10 Å graph
    # --------------------------------------------------------

    dist_matrix = torch.cdist(pos, pos)

    edge_index = (
        dist_matrix < GRAPH_CUTOFF
    ).nonzero(
        as_tuple=False
    ).t()

    if edge_index.shape[1] > 0:
        edge_index = edge_index[
            :,
            edge_index[0] != edge_index[1],
        ]

    if edge_index.shape[1] == 0:
        raise ValueError(
            "目标链没有构建出任何 graph edge"
        )

    # --------------------------------------------------------
    # 5维 Gaussian distance features
    # --------------------------------------------------------

    dist_val = dist_matrix[
        edge_index[0],
        edge_index[1],
    ]

    scales = torch.tensor(
        [1.0, 2.0, 5.0, 10.0, 20.0],
        dtype=dist_val.dtype,
        device=dist_val.device,
    )

    edge_attr = torch.exp(
        -0.5
        *
        (
            dist_val.unsqueeze(-1)
            /
            scales
        ) ** 2
    )

    if edge_attr.shape[1] != EDGE_DIM:
        raise ValueError(
            f"edge_attr dimension = {edge_attr.shape[1]}，expected {EDGE_DIM}"
        )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
    )

    return data, res_mapping_info


# ============================================================
# 5. 自动由参考复合物生成 true interface residues
#
# 正式定义：
# target residue 与 partner chain(s) 任一非氢原子距离 < 5 Å
# ============================================================

def get_true_interface_from_reference_complex(
        reference_pdb,
        target_chain,
        partner_chain_string,
        cutoff=5.0):

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure(
        "reference_complex",
        reference_pdb,
    )

    model = structure[0]

    if target_chain not in model:
        raise ValueError(
            f"参考复合物中不存在 target chain '{target_chain}'"
        )

    partner_chains = parse_chain_ids(partner_chain_string)

    if not partner_chains:
        raise ValueError("没有提供 partner chain")

    for chain_id in partner_chains:
        if chain_id not in model:
            raise ValueError(
                f"参考复合物中不存在 partner chain '{chain_id}'"
            )

    partner_heavy_atoms = []

    for chain_id in partner_chains:
        for residue in model[chain_id]:
            if not is_standard_residue(residue):
                continue

            for atom in residue:
                if is_heavy_atom(atom):
                    partner_heavy_atoms.append(atom)

    if not partner_heavy_atoms:
        raise ValueError(
            "partner chains 中未找到有效非氢原子"
        )

    neighbor_search = NeighborSearch(partner_heavy_atoms)

    true_interface = set()

    for residue in model[target_chain]:
        if not is_standard_residue(residue):
            continue

        uid = residue_uid(residue)
        is_interface = False

        for atom in residue:
            if not is_heavy_atom(atom):
                continue

            neighbors = neighbor_search.search(
                atom.coord,
                cutoff,
                level="A",
            )

            # NeighborSearch 先找 cutoff 内候选，再严格检查 < cutoff。
            for partner_atom in neighbors:
                distance = np.linalg.norm(
                    atom.coord - partner_atom.coord
                )

                if distance < cutoff:
                    is_interface = True
                    break

            if is_interface:
                break

        if is_interface:
            true_interface.add(uid)

    return true_interface


# ============================================================
# 6. Top-K ranking evaluation
# ============================================================

def add_ground_truth_and_rank(
        df_results,
        true_interface_residues):

    """
    对全部扫描残基：
    1. 加入5 Å ground truth；
    2. 按预测Probability降序；
    3. 从1开始赋予Rank。

    不进行任何classification-threshold筛选。
    """

    df = df_results.copy()

    true_set = {
        str(x)
        for x in true_interface_residues
    }

    df["Ground_Truth"] = (
        df["Residue_UID"]
        .astype(str)
        .isin(true_set)
        .astype(int)
    )

    df = (
        df
        .sort_values(
            by="Probability",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    df.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(df) + 1
        )
    )

    return df


def calculate_overall_ranking_metrics(
        ranked_df):

    """
    AUROC和AUPRC使用全部扫描残基的连续预测概率，
    与Top-K选择无关。
    """

    y_true = (
        ranked_df[
            "Ground_Truth"
        ]
        .to_numpy(
            dtype=int
        )
    )

    y_prob = (
        ranked_df[
            "Probability"
        ]
        .to_numpy(
            dtype=float
        )
    )

    n_total = int(
        len(
            ranked_df
        )
    )

    n_true = int(
        np.sum(
            y_true == 1
        )
    )

    prevalence = (
        float(
            n_true / n_total
        )
        if n_total > 0
        else np.nan
    )

    if len(
        np.unique(
            y_true
        )
    ) == 2:

        auroc = float(
            roc_auc_score(
                y_true,
                y_prob
            )
        )

        auprc = float(
            average_precision_score(
                y_true,
                y_prob
            )
        )

    else:

        auroc = np.nan
        auprc = np.nan


    return {
        "Total_scanned_residues":
            n_total,

        "Total_true_interface_residues":
            n_true,

        "Interface_prevalence":
            prevalence,

        "AUROC":
            auroc,

        "AUPRC":
            auprc,
    }


def calculate_topk_metrics(
        ranked_df,
        top_k_values):

    """
    对固定Top-K计算：
        TP@K
        FP@K
        FN@K
        Precision@K
        Recall@K
        EnrichmentFactor@K

    Enrichment factor:
        Precision@K / overall interface prevalence

    这里Top-K表示“候选残基优先排序”，
    不是classification threshold。
    """

    total_true = int(
        ranked_df[
            "Ground_Truth"
        ].sum()
    )

    total_residues = int(
        len(
            ranked_df
        )
    )

    prevalence = (
        total_true / total_residues
        if total_residues > 0
        else np.nan
    )

    rows = []

    for requested_k in top_k_values:

        actual_k = min(
            int(
                requested_k
            ),
            total_residues
        )

        top_df = ranked_df.head(
            actual_k
        )

        tp = int(
            top_df[
                "Ground_Truth"
            ].sum()
        )

        fp = int(
            actual_k - tp
        )

        fn = int(
            total_true - tp
        )

        precision_at_k = (
            tp / actual_k
            if actual_k > 0
            else np.nan
        )

        recall_at_k = (
            tp / total_true
            if total_true > 0
            else np.nan
        )

        enrichment_factor = (
            precision_at_k / prevalence
            if (
                prevalence is not None
                and
                not np.isnan(
                    prevalence
                )
                and
                prevalence > 0
            )
            else np.nan
        )


        true_residues = (
            top_df.loc[
                top_df[
                    "Ground_Truth"
                ] == 1,
                "Residue_UID"
            ]
            .astype(str)
            .tolist()
        )

        false_residues = (
            top_df.loc[
                top_df[
                    "Ground_Truth"
                ] == 0,
                "Residue_UID"
            ]
            .astype(str)
            .tolist()
        )


        rows.append({

            "K":
                int(
                    requested_k
                ),

            "Actual_K":
                actual_k,

            "TP_at_K":
                tp,

            "FP_at_K":
                fp,

            "FN_at_K":
                fn,

            "Precision_at_K":
                float(
                    precision_at_k
                ),

            "Recall_at_K":
                float(
                    recall_at_k
                ),

            "Enrichment_Factor_at_K":
                float(
                    enrichment_factor
                )
                if not np.isnan(
                    enrichment_factor
                )
                else np.nan,

            "True_interface_residues_in_TopK":
                ",".join(
                    true_residues
                ),

            "Noninterface_residues_in_TopK":
                ",".join(
                    false_residues
                ),
        })


    return pd.DataFrame(
        rows
    )


def add_distance_to_true_interface(
        ranked_df):

    """
    计算每个残基C-alpha到最近真实界面残基C-alpha的距离。

    该值只用于帮助解释Top-K中的非界面候选是否位于界面附近，
    不改变5 Å heavy-atom ground truth，也不改变TP/FP定义。
    """

    df = ranked_df.copy()

    true_coords = []

    for _, row in df[
        df[
            "Ground_Truth"
        ] == 1
    ].iterrows():

        true_coords.append(
            np.asarray(
                [
                    row["CA_X"],
                    row["CA_Y"],
                    row["CA_Z"],
                ],
                dtype=float,
            )
        )


    min_distances = []

    for _, row in df.iterrows():

        if not true_coords:

            min_distances.append(
                np.nan
            )

            continue


        coord = np.asarray(
            [
                row["CA_X"],
                row["CA_Y"],
                row["CA_Z"],
            ],
            dtype=float,
        )


        distances = [
            np.linalg.norm(
                coord
                -
                true_coord
            )
            for true_coord
            in true_coords
        ]


        min_distances.append(
            float(
                min(
                    distances
                )
            )
        )


    df[
        "Min_CA_Distance_to_True_Interface_A"
    ] = min_distances


    return df


# ============================================================
# 7. 推理主流程
# ============================================================

def run_prediction(
        pdb_file,
        target_chain,
        ground_truth_residues=None):

    """
    使用原正式5A模型和原预测特征流程生成全部残基概率。

    重要：
    - 不使用0.2、0.01或其他阈值筛选残基。
    - 全部实际扫描残基均保留。
    - 评价采用Probability ranking + Top-K。
    """

    print(
        "\nLoading ESM-2..."
    )


    model_esm, alphabet = (
        esm.pretrained.esm2_t33_650M_UR50D()
    )

    model_esm = (
        model_esm
        .to(
            DEVICE
        )
        .eval()
    )

    batch_converter = (
        alphabet.get_batch_converter()
    )


    print(
        "Loading old formal 5A GNN weights..."
    )


    model = PPI_Model(
        in_dim=TOTAL_NODE_DIM,
        edge_dim=EDGE_DIM,
    ).to(
        DEVICE
    )


    try:

        state_dict = torch.load(
            WEIGHTS_PATH,
            map_location=DEVICE,
        )

        model.load_state_dict(
            state_dict
        )


    except Exception as e:

        raise RuntimeError(
            f"权重加载失败：{e}"
        )


    model.eval()


    print(
        f"\nBuilding graph for chain "
        f"[{target_chain}]..."
    )


    graph_data, res_info = (
        build_graph_for_prediction(
            pdb_file,
            target_chain,
            model_esm,
            batch_converter,
        )
    )


    graph_data = graph_data.to(
        DEVICE
    )


    with torch.no_grad():

        logits = model(
            graph_data
        )

        probs = (
            torch.sigmoid(
                logits
            )
            .detach()
            .cpu()
            .numpy()
        )


    all_predictions = []


    for i, info in enumerate(
        res_info
    ):

        probability = float(
            probs[
                i
            ]
        )

        coord = info[
            "coord"
        ]


        all_predictions.append({

            "Chain":
                info[
                    "chain"
                ],

            "Res_Name":
                info[
                    "res_name"
                ],

            "PDB_ID":
                info[
                    "pdb_id"
                ],

            "Insertion_Code":
                info[
                    "insertion_code"
                ],

            "Residue_UID":
                info[
                    "residue_uid"
                ],

            "Probability":
                probability,

            "CA_X":
                float(
                    coord[
                        0
                    ]
                ),

            "CA_Y":
                float(
                    coord[
                        1
                    ]
                ),

            "CA_Z":
                float(
                    coord[
                        2
                    ]
                ),
        })


    if not all_predictions:

        raise RuntimeError(
            "没有生成任何预测结果"
        )


    df_results = pd.DataFrame(
        all_predictions
    )


    # ========================================================
    # 全部残基直接按Probability降序
    # ========================================================

    df_ranked_no_truth = (
        df_results
        .sort_values(
            by="Probability",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    df_ranked_no_truth.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(
                df_ranked_no_truth
            ) + 1
        )
    )


    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "Prediction finished | "
        "ALL scanned residues retained"
    )

    print(
        f"Target chain: "
        f"{target_chain}"
    )

    print(
        f"Scanned residues: "
        f"{len(df_results)}"
    )

    print(
        "Ranking criterion: "
        "Probability descending"
    )

    print(
        "=" * 72
    )


    # ========================================================
    # 打印Top 30
    # ========================================================

    print(
        "\nTop 30 ranked residues:"
    )

    print(
        f"{'Rank':<7}"
        f"{'Chain':<7}"
        f"{'Residue':<10}"
        f"{'PDB ID':<10}"
        f"{'Probability':<12}"
    )

    print(
        "-" * 50
    )


    for _, row in (
        df_ranked_no_truth
        .head(
            30
        )
        .iterrows()
    ):

        print(
            f"{int(row['Rank']):<7}"
            f"{row['Chain']:<7}"
            f"{row['Res_Name']:<10}"
            f"{str(row['Residue_UID']):<10}"
            f"{row['Probability']:<12.6f}"
        )


    base_name = (
        os.path.basename(
            pdb_file
        )
        .replace(
            ".pdb",
            ""
        )
    )


    # ========================================================
    # 无ground truth时：仍保存全部排序结果
    # ========================================================

    if ground_truth_residues is None:

        ranked_csv = os.path.join(
            OUTPUT_DIR,
            f"{base_name}_chain_{target_chain}_ALL_residues_ranked.csv",
        )


        df_ranked_no_truth.to_csv(
            ranked_csv,
            index=False,
            encoding="utf-8-sig",
        )


        print(
            "\nNo ground truth supplied."
        )

        print(
            "Ranked prediction CSV saved:"
        )

        print(
            ranked_csv
        )


        return


    # ========================================================
    # Ground-truth evaluation
    # ========================================================

    ground_truth_residues = {
        str(
            x
        )
        for x
        in ground_truth_residues
    }


    scanned_uid_set = set(
        df_results[
            "Residue_UID"
        ]
        .astype(
            str
        )
        .tolist()
    )


    truth_not_scanned = sorted(
        ground_truth_residues
        -
        scanned_uid_set,
        key=lambda x: (
            int(
                re.match(
                    r"-?\d+",
                    x
                ).group()
            )
            if re.match(
                r"-?\d+",
                x
            )
            else 0,
            x,
        )
    )


    if truth_not_scanned:

        print(
            "\nWARNING: "
            "下列ground-truth residues不在模型实际扫描区域内："
        )

        print(
            ",".join(
                truth_not_scanned
            )
        )

        print(
            "这些残基不进入当前Top-K Recall计算；"
            "请检查1022 aa截断或编号对应关系。"
        )


    evaluated_truth = (
        ground_truth_residues
        &
        scanned_uid_set
    )


    ranked_df = (
        add_ground_truth_and_rank(
            df_results,
            evaluated_truth,
        )
    )


    ranked_df = (
        add_distance_to_true_interface(
            ranked_df
        )
    )


    overall_metrics = (
        calculate_overall_ranking_metrics(
            ranked_df
        )
    )


    overall_metrics[
        "N_ground_truth_residues_outside_scanned_region"
    ] = len(
        truth_not_scanned
    )


    overall_metrics[
        "Ground_truth_residues_outside_scanned_region"
    ] = ",".join(
        truth_not_scanned
    )


    topk_df = (
        calculate_topk_metrics(
            ranked_df,
            TOP_K_VALUES,
        )
    )


    # ========================================================
    # Terminal summary
    # ========================================================

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "CR1 CASE STUDY — PROBABILITY RANKING / TOP-K EVALUATION"
    )

    print(
        "=" * 72
    )


    print(
        "Total scanned residues:",
        overall_metrics[
            "Total_scanned_residues"
        ],
    )

    print(
        "True interface residues in scanned region:",
        overall_metrics[
            "Total_true_interface_residues"
        ],
    )

    print(
        "Interface prevalence:",
        f"{overall_metrics['Interface_prevalence']:.4f}",
    )


    if not pd.isna(
        overall_metrics[
            "AUROC"
        ]
    ):

        print(
            f"AUROC = "
            f"{overall_metrics['AUROC']:.4f}"
        )

        print(
            f"AUPRC = "
            f"{overall_metrics['AUPRC']:.4f}"
        )


    print(
        "\nTop-K summary:"
    )

    print(
        f"{'K':<6}"
        f"{'TP':<7}"
        f"{'FP':<7}"
        f"{'FN':<7}"
        f"{'Precision@K':<15}"
        f"{'Recall@K':<13}"
        f"{'EF@K':<10}"
    )

    print(
        "-" * 68
    )


    for _, row in topk_df.iterrows():

        print(
            f"{int(row['K']):<6}"
            f"{int(row['TP_at_K']):<7}"
            f"{int(row['FP_at_K']):<7}"
            f"{int(row['FN_at_K']):<7}"
            f"{row['Precision_at_K']:<15.4f}"
            f"{row['Recall_at_K']:<13.4f}"
            f"{row['Enrichment_Factor_at_K']:<10.4f}"
        )


    # ========================================================
    # 特别打印Top-8，用于原Figure 7对应分析
    # ========================================================

    top8_df = ranked_df.head(
        min(
            8,
            len(
                ranked_df
            )
        )
    )


    print(
        "\n"
        +
        "-" * 72
    )

    print(
        "TOP-8 RESIDUES — corresponding to original Figure 7 ranking"
    )

    print(
        "-" * 72
    )


    print(
        f"{'Rank':<7}"
        f"{'Residue':<10}"
        f"{'PDB ID':<10}"
        f"{'Probability':<13}"
        f"{'5A Truth':<10}"
        f"{'Min CA dist':<12}"
    )


    for _, row in top8_df.iterrows():

        print(
            f"{int(row['Rank']):<7}"
            f"{row['Res_Name']:<10}"
            f"{str(row['Residue_UID']):<10}"
            f"{row['Probability']:<13.6f}"
            f"{int(row['Ground_Truth']):<10}"
            f"{row['Min_CA_Distance_to_True_Interface_A']:<12.3f}"
        )


    top8_summary = topk_df[
        topk_df[
            "K"
        ] == 8
    ]


    if not top8_summary.empty:

        row8 = top8_summary.iloc[
            0
        ]

        print(
            "\nFigure 7 Top-8 quantitative result:"
        )

        print(
            f"TP@8 = "
            f"{int(row8['TP_at_K'])}"
        )

        print(
            f"FP@8 = "
            f"{int(row8['FP_at_K'])}"
        )

        print(
            f"FN@8 = "
            f"{int(row8['FN_at_K'])}"
        )

        print(
            f"Precision@8 = "
            f"{row8['Precision_at_K']:.4f}"
        )

        print(
            f"Recall@8 = "
            f"{row8['Recall_at_K']:.4f}"
        )

        print(
            f"Enrichment factor@8 = "
            f"{row8['Enrichment_Factor_at_K']:.4f}"
        )

        print(
            "True interface residues in Top-8:",
            (
                row8[
                    "True_interface_residues_in_TopK"
                ]
                if row8[
                    "True_interface_residues_in_TopK"
                ]
                else "None"
            ),
        )


    print(
        "=" * 72
    )


    # ========================================================
    # Save outputs
    # ========================================================

    ranked_csv = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_chain_{target_chain}_ALL_residues_ranked_with_5A_truth.csv",
    )


    topk_csv = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_chain_{target_chain}_TopK_summary.csv",
    )


    overall_csv = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_chain_{target_chain}_overall_ranking_metrics.csv",
    )


    top8_csv = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_chain_{target_chain}_Top8_Figure7.csv",
    )


    ranked_df.to_csv(
        ranked_csv,
        index=False,
        encoding="utf-8-sig",
    )


    topk_df.to_csv(
        topk_csv,
        index=False,
        encoding="utf-8-sig",
    )


    pd.DataFrame(
        [
            overall_metrics
        ]
    ).to_csv(
        overall_csv,
        index=False,
        encoding="utf-8-sig",
    )


    top8_df.to_csv(
        top8_csv,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        "\nSaved:"
    )

    print(
        ranked_csv
    )

    print(
        topk_csv
    )

    print(
        overall_csv
    )

    print(
        top8_csv
    )


# ============================================================
# 8. Entry
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Prediction PDB
    # --------------------------------------------------------

    while True:

        pdb_input = input(
            "\n1. 请输入待预测 PDB 文件路径: "
        ).strip().strip('"')


        if os.path.exists(
            pdb_input
        ):

            break


        print(
            "找不到该 PDB，请重新输入。"
        )


    # --------------------------------------------------------
    # 2. Target chain
    # --------------------------------------------------------

    target_chain = input(
        "2. 请输入目标链 ID（单链，如 C）: "
    ).strip()


    if len(
        target_chain
    ) != 1:

        raise ValueError(
            "当前 CR1 Top-K case evaluation "
            "要求 target_chain 为单个 chain ID。"
        )


    # --------------------------------------------------------
    # 3. Ground truth mode
    # --------------------------------------------------------

    print(
        "\n本脚本不设置分类阈值："
    )

    print(
        "全部扫描残基都会保留并按Probability降序排序。"
    )

    print(
        "固定评价 Top-5 / Top-8 / Top-10 / Top-20 / Top-30。"
    )


    print(
        "\nGround-truth mode:"
    )

    print(
        "  0 = 不做ground-truth评价，只输出全部残基降序排名"
    )

    print(
        "  1 = 手动输入全部真实界面残基编号"
    )

    print(
        "  2 = 从参考复合物PDB按非氢原子距离 <5 Å自动生成"
    )


    gt_mode = input(
        "3. 请选择 ground-truth mode [0/1/2]: "
    ).strip()


    ground_truth_residues = None


    # --------------------------------------------------------
    # Mode 1: manual
    # --------------------------------------------------------

    if gt_mode == "1":

        truth_input = input(
            "请输入该目标链全部真实界面残基编号"
            "（如 91,140,141,1114）: "
        ).strip()


        ground_truth_residues = (
            parse_manual_ground_truth(
                truth_input
            )
        )


        if not ground_truth_residues:

            raise ValueError(
                "没有解析到任何 ground-truth residue。"
            )


        print(
            "Manual ground truth:",
            ",".join(
                sorted(
                    ground_truth_residues
                )
            ),
        )


    # --------------------------------------------------------
    # Mode 2: automatic 5 Å definition
    # --------------------------------------------------------

    elif gt_mode == "2":

        while True:

            reference_pdb = input(
                "请输入用于 ground truth 的参考复合物 PDB 路径: "
            ).strip().strip('"')


            if os.path.exists(
                reference_pdb
            ):

                break


            print(
                "找不到参考复合物 PDB，请重新输入。"
            )


        reference_target_chain = input(
            f"参考复合物中的目标链 ID "
            f"(直接回车使用 {target_chain}): "
        ).strip()


        if not reference_target_chain:

            reference_target_chain = (
                target_chain
            )


        partner_chains = input(
            "请输入参考复合物中的 partner chain(s) "
            "（例如 A、AB 或 A,B）: "
        ).strip()


        ground_truth_residues = (
            get_true_interface_from_reference_complex(
                reference_pdb,
                reference_target_chain,
                partner_chains,
                cutoff=GROUND_TRUTH_CONTACT_CUTOFF,
            )
        )


        print(
            "\nAutomatically generated "
            f"{len(ground_truth_residues)} "
            f"true interface residues "
            f"using heavy-atom distance < "
            f"{GROUND_TRUTH_CONTACT_CUTOFF:.1f} Å."
        )


        print(
            "Ground-truth residues:"
        )


        print(
            ",".join(
                sorted(
                    ground_truth_residues,
                    key=lambda x: (
                        int(
                            re.match(
                                r"-?\d+",
                                x
                            ).group()
                        )
                        if re.match(
                            r"-?\d+",
                            x
                        )
                        else 0,
                        x,
                    ),
                )
            )
        )


    elif (
        gt_mode == "0"
        or
        gt_mode == ""
    ):

        ground_truth_residues = (
            None
        )


    else:

        raise ValueError(
            "ground-truth mode 只能是 0、1 或 2。"
        )


    # --------------------------------------------------------
    # 4. Run
    # --------------------------------------------------------

    run_prediction(
        pdb_input,
        target_chain,
        ground_truth_residues=ground_truth_residues,
    )
