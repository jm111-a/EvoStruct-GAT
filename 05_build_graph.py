import os
import random
import warnings
import tempfile

import torch
import esm
import numpy as np
import pandas as pd

from tqdm import tqdm
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.DSSP import DSSP
from Bio.SeqUtils import seq1
from torch_geometric.data import Data


warnings.filterwarnings("ignore")


# ============================================================
# 1. 路径配置
# ============================================================

ROOT = r"C:\Users\Administrator\Desktop\P-P"


# 所有1739个处理后的PDB统一放在这里
PDB_DIR = os.path.join(
    ROOT,
    "03_chain_trimmed_pdb"
)


# 新cluster split
SPLIT_DIR = os.path.join(
    ROOT,
    "cluster_split_v3_A2",
    "04_split"
)


# 已经生成完成的5 Å labels
LABEL_DIR = os.path.join(
    ROOT,
    "cluster_split_v3_A2",
    "05_labels"
)


# ============================================================
# 新graph输出目录
#
# IMPORTANT:
# 这里使用全新的目录，绝不覆盖旧的：
# cluster_split_v3_A2/06_graphs
# 或 sensitivity_analysis/5A/06_graphs
#
# 本版本唯一的结构特征变化：
# DSSP/RSA只在当前 interaction partner 上计算。
# 如果一个partner由多条链组成（如 receptor=HL），
# H和L会一起保留并共同计算DSSP；对侧partner完全移除。
# ============================================================

OUTPUT_BASE = os.path.join(
    ROOT,
    "cluster_split_v3_A2",
    "partner_only_DSSP",
    "06_graphs"
)


# DSSP
DSSP_EXE_PATH = os.path.join(
    ROOT,
    "dssp.exe"
)


# ============================================================
# corrected 5A正式数据应保持不变的graph/node/positive数量
#
# 本次只改变DSSP/RSA的计算环境，因此：
# - graph数量不应改变
# - node数量不应改变
# - 5 Å标签数量不应改变
# ============================================================

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
# 2. 设备
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if torch.cuda.is_available():
    torch.cuda.empty_cache()


print("=" * 70)
print("EvoStruct-GAT graph construction | partner-only DSSP/RSA")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"DSSP: {DSSP_EXE_PATH}")
print(f"Output: {OUTPUT_BASE}")


if not os.path.exists(DSSP_EXE_PATH):
    raise FileNotFoundError(
        f"找不到DSSP程序: {DSSP_EXE_PATH}"
    )


# ============================================================
# 3. 随机种子
# ============================================================

def set_all_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_all_seed(42)


# ============================================================
# 4. 加载原始论文使用的 ESM-2
# ============================================================

print(
    f"Loading esm2_t33_650M_UR50D on {DEVICE}..."
)

model_esm, alphabet = (
    esm.pretrained.esm2_t33_650M_UR50D()
)

model_esm = (
    model_esm
    .to(DEVICE)
    .eval()
)

batch_converter = (
    alphabet.get_batch_converter()
)

print("ESM-2 loaded successfully.")


# ============================================================
# 5. chain ID解析
# ============================================================

def parse_chain_ids(chain_ids_str):

    """
    例如：
    A    -> ['A']
    HL   -> ['H', 'L']
    A,B  -> ['A', 'B']
    """

    s = str(chain_ids_str)

    s = (
        s
        .replace(",", "")
        .replace(" ", "")
    )

    return list(s)


# ============================================================
# 6. target-partner-only DSSP辅助工具
# ============================================================

class KeepTargetChains(Select):

    """
    PDBIO选择器：
    只保留当前interaction partner包含的chain。

    例如：
        receptor = HL
        ligand   = A

    构建receptor graph时：
        临时PDB只保留H和L，
        DSSP在H+L这个完整partner上计算。

    构建ligand graph时：
        临时PDB只保留A，
        DSSP只在A上计算。
    """

    def __init__(self, target_chain_ids):

        super().__init__()

        self.target_chain_ids = set(
            target_chain_ids
        )


    def accept_chain(
            self,
            chain
    ):

        return (
            1
            if chain.id
            in self.target_chain_ids
            else 0
        )


def calculate_partner_only_dssp(
        model,
        target_chain_ids,
        pdb_id,
        chain_type,
        parser
):

    """
    生成仅包含当前interaction partner的临时PDB，
    并在该临时PDB上运行DSSP。

    重要：
    1. 不改变原始03_chain_trimmed_pdb。
    2. 不改变坐标。
    3. 多链partner作为整体保留。
    4. 对侧interaction partner完全不参与DSSP/RSA计算。
    5. 临时PDB在DSSP结束后立即删除。
    """

    temp_pdb_path = None

    try:

        # ----------------------------------------------------
        # Windows下先创建临时文件名，然后关闭句柄
        # ----------------------------------------------------

        fd, temp_pdb_path = tempfile.mkstemp(
            prefix=f"{pdb_id}_{chain_type}_partner_",
            suffix=".pdb"
        )

        os.close(
            fd
        )


        # ----------------------------------------------------
        # 将当前target partner写入临时PDB
        # ----------------------------------------------------

        io = PDBIO()

        io.set_structure(
            model
        )

        io.save(
            temp_pdb_path,
            KeepTargetChains(
                target_chain_ids
            )
        )


        # ----------------------------------------------------
        # 重新读取临时target-partner结构
        #
        # DSSP使用的model必须和传入的临时PDB相对应。
        # ----------------------------------------------------

        partner_structure = parser.get_structure(
            f"{pdb_id}_{chain_type}_partner",
            temp_pdb_path
        )

        partner_model = partner_structure[
            0
        ]


        # ----------------------------------------------------
        # DSSP只在当前interaction partner上计算
        # ----------------------------------------------------

        dssp_dict = dict(
            DSSP(
                partner_model,
                temp_pdb_path,
                dssp=DSSP_EXE_PATH
            )
        )


        return dssp_dict


    finally:

        # ----------------------------------------------------
        # 无论DSSP成功或失败，都删除临时PDB
        # ----------------------------------------------------

        if (
            temp_pdb_path is not None
            and
            os.path.exists(
                temp_pdb_path
            )
        ):

            try:

                os.remove(
                    temp_pdb_path
                )

            except Exception:

                pass


# ============================================================
# 7. 单侧蛋白建图
# ============================================================

def process_chain_graph(
        pdb_path,
        chain_ids_str,
        labels_df,
        out_dirs,
        chain_type,
        pdb_id
):

    parser = PDBParser(
        QUIET=True
    )

    structure = parser.get_structure(
        pdb_id,
        pdb_path
    )

    model = structure[0]


    # --------------------------------------------------------
    # target side链
    # --------------------------------------------------------

    target_chain_ids = parse_chain_ids(
        chain_ids_str
    )

    all_residues = []


    for c_id in target_chain_ids:

        if c_id not in model:

            raise ValueError(
                f"{pdb_id}: 找不到target chain {c_id}"
            )

        for res in model[c_id]:

            if res.id[0] == " ":

                all_residues.append(
                    res
                )


    if not all_residues:

        raise ValueError(
            f"{pdb_id}: target side没有有效残基"
        )


    # ========================================================
    # DSSP / RSA
    #
    # IMPORTANT:
    # 旧代码在完整receptor-ligand复合物上运行DSSP。
    #
    # 现在改为：
    # 只保留当前interaction partner，再运行DSSP。
    #
    # 因此：
    # receptor graph -> receptor partner only
    # ligand graph   -> ligand partner only
    #
    # 多链partner会整体保留。
    # ========================================================

    dssp_dict = None

    try:

        dssp_dict = calculate_partner_only_dssp(

            model=model,

            target_chain_ids=target_chain_ids,

            pdb_id=pdb_id,

            chain_type=chain_type,

            parser=parser

        )

    except Exception as e:

        print(
            f"\nWARNING partner-only DSSP failed "
            f"({pdb_id}, {chain_type}): {e}"
        )


    # ========================================================
    # 构建target-side sequence
    # ========================================================

    seq = ""

    for res in all_residues:

        try:

            aa = seq1(
                res.get_resname(),
                custom_map={
                    "MSE": "M"
                }
            )

        except Exception:

            aa = "X"


        if aa in ["", "?"]:
            aa = "X"

        seq += aa


    # ESM-2最大长度
    if len(seq) > 1022:

        seq = seq[:1022]


    # ========================================================
    # ESM-2
    # ========================================================

    data_esm = [
        (
            pdb_id,
            seq
        )
    ]


    _, _, batch_tokens = batch_converter(
        data_esm
    )

    batch_tokens = batch_tokens.to(
        DEVICE
    )


    with torch.no_grad():

        results = model_esm(
            batch_tokens,
            repr_layers=[33],
            return_contacts=False
        )


        token_representations = (
            results[
                "representations"
            ][33][
                0,
                1:len(seq) + 1
            ]
            .cpu()
            .numpy()
        )


    # ========================================================
    # 当前target side对应的label
    # ========================================================

    relevant_labels = (
        labels_df[
            labels_df[
                "chain_id"
            ].isin(
                target_chain_ids
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # corrected 5A label mapping QC
    #
    # 旧版按：
    #     chain_id + residue_index
    # 查找标签。
    #
    # 对存在PDB insertion code且residue_index重复的情况，
    # 这种做法可能把同一编号的多个残基映射到第一条记录。
    #
    # 现在使用：
    #     target-side residue order
    #     ==
    #     label CSV row order
    #
    # 并逐节点严格检查Chain / Res_ID / Res_Name。
    # ========================================================

    if len(
        relevant_labels
    ) != len(
        all_residues
    ):

        raise ValueError(
            f"{pdb_id}_{chain_type}: "
            f"target residues={len(all_residues)}, "
            f"label rows={len(relevant_labels)}，"
            f"数量不一致"
        )


    node_features = []

    node_coords = []

    node_labels = []

    csv_rows = []


    # 如果序列超过1022，只处理前1022残基
    valid_count = min(
        len(all_residues),
        len(token_representations)
    )


    # ========================================================
    # 节点特征
    # ========================================================

    for i in range(
        valid_count
    ):

        res = all_residues[i]

        res_idx = res.get_id()[1]

        c_id = res.get_full_id()[2]


        # ----------------------------------------------------
        # corrected label匹配：
        # 按target-side节点顺序一一对应
        # ----------------------------------------------------

        label_row = relevant_labels.iloc[
            i
        ]


        label_chain = str(
            label_row[
                "chain_id"
            ]
        )


        try:

            label_res_idx = int(
                label_row[
                    "residue_index"
                ]
            )

        except Exception:

            label_res_idx = label_row[
                "residue_index"
            ]


        label_res_name = str(
            label_row[
                "residue_name"
            ]
        ).upper()


        current_res_name = str(
            res.get_resname()
        ).upper()


        # ----------------------------------------------------
        # 严格QC：
        # 顺序对应的chain / residue number / residue name
        # 必须一致。
        #
        # insertion code不再依赖residue_index作为唯一键，
        # 而是通过原始顺序保证一一对应。
        # ----------------------------------------------------

        if (
            label_chain != c_id
            or
            label_res_idx != res_idx
            or
            label_res_name != current_res_name
        ):

            raise ValueError(
                f"{pdb_id}_{chain_type}: "
                f"label-order mismatch at node {i}: "
                f"PDB=({c_id},{res_idx},{current_res_name}), "
                f"Label=({label_chain},"
                f"{label_res_idx},{label_res_name})"
            )


        y_val = int(
            label_row[
                "interface_label"
            ]
        )


        # ----------------------------------------------------
        # 默认DSSP特征
        # ----------------------------------------------------

        struct_feat = [
            0.0,
            0.0,
            1.0,
            0.5
        ]

        sec_struct_char = "-"


        # ----------------------------------------------------
        # DSSP真实特征
        # ----------------------------------------------------

        if dssp_dict is not None:

            dssp_key = (
                c_id,
                res.get_id()
            )


            if dssp_key in dssp_dict:

                dssp_val = (
                    dssp_dict[
                        dssp_key
                    ]
                )

                sec_struct_char = (
                    dssp_val[2]
                )

                rsa = dssp_val[3]


                # H/G/I -> alpha
                is_H = (
                    1.0
                    if sec_struct_char
                    in ["H", "G", "I"]
                    else 0.0
                )


                # B/E -> beta
                is_E = (
                    1.0
                    if sec_struct_char
                    in ["B", "E"]
                    else 0.0
                )


                # 其他 -> coil
                is_C = (
                    0.0
                    if (
                        is_H == 1.0
                        or
                        is_E == 1.0
                    )
                    else 1.0
                )


                try:

                    rsa = float(
                        rsa
                    )

                except (
                    ValueError,
                    TypeError
                ):

                    rsa = 0.5


                struct_feat = [
                    is_H,
                    is_E,
                    is_C,
                    rsa
                ]


        # ----------------------------------------------------
        # ESM1280 + DSSP4 = 1284
        # ----------------------------------------------------

        full_feat = np.concatenate(
            [
                token_representations[i],
                struct_feat
            ]
        )


        # ----------------------------------------------------
        # C-alpha coordinate
        # ----------------------------------------------------

        if "CA" in res:

            coord = (
                res[
                    "CA"
                ].get_coord()
            )

        else:

            coord = (
                res.child_list[
                    0
                ].get_coord()
            )


        node_features.append(
            full_feat
        )

        node_coords.append(
            coord
        )

        node_labels.append(
            y_val
        )


        csv_rows.append({

            "PDB_ID":
                pdb_id,

            "Chain":
                c_id,

            "Res_Name":
                res.get_resname(),

            "Res_ID":
                res_idx,

            "Coord_X":
                round(
                    float(coord[0]),
                    2
                ),

            "Coord_Y":
                round(
                    float(coord[1]),
                    2
                ),

            "Coord_Z":
                round(
                    float(coord[2]),
                    2
                ),

            "Sec_Struct":
                sec_struct_char,

            "Is_Helix":
                struct_feat[0],

            "Is_Sheet":
                struct_feat[1],

            "Is_Coil":
                struct_feat[2],

            "RSA":
                round(
                    float(
                        struct_feat[3]
                    ),
                    4
                ),

            "Label":
                y_val
        })


    if len(node_features) < 5:

        raise ValueError(
            f"{pdb_id}_{chain_type}: "
            f"有效节点数不足5"
        )


    # ========================================================
    # Tensor
    # ========================================================

    x = torch.tensor(
        np.array(
            node_features
        ),
        dtype=torch.float
    )


    pos = torch.tensor(
        np.array(
            node_coords
        ),
        dtype=torch.float
    )


    y = torch.tensor(
        np.array(
            node_labels
        ),
        dtype=torch.long
    )


    # ========================================================
    # C-alpha 10 Å graph
    # ========================================================

    dist_matrix = torch.cdist(
        pos,
        pos
    )


    edge_index = (
        dist_matrix < 10.0
    ).nonzero(
        as_tuple=False
    ).t()


    # --------------------------------------------------------
    # 删除self-loop
    #
    # 注意：
    # edge_index shape = [2, E]
    # 所以mask必须作用于第二维
    # --------------------------------------------------------

    self_loop_mask = (
        edge_index[0]
        !=
        edge_index[1]
    )


    edge_index = edge_index[
        :,
        self_loop_mask
    ]


    if edge_index.shape[1] == 0:

        raise ValueError(
            f"{pdb_id}_{chain_type}: "
            "没有构建出任何graph edge"
        )


    # ========================================================
    # 5维Gaussian RBF edge features
    # ========================================================

    dist_val = dist_matrix[
        edge_index[0],
        edge_index[1]
    ]


    scales = torch.tensor(
        [
            1.0,
            2.0,
            5.0,
            10.0,
            20.0
        ],
        dtype=torch.float,
        device=dist_val.device
    )


    edge_attr = torch.exp(

        -0.5
        *
        (
            dist_val.unsqueeze(
                -1
            )
            /
            scales
        ) ** 2

    )


    # ========================================================
    # PyG Data
    # ========================================================

    data = Data(

        x=x,

        edge_index=edge_index,

        edge_attr=edge_attr,

        y=y,

        pos=pos
    )


    # ========================================================
    # 节点degree
    # ========================================================

    node_degrees = (
        torch.bincount(
            edge_index[0],
            minlength=len(
                node_features
            )
        )
        .cpu()
        .numpy()
    )


    for i in range(
        len(csv_rows)
    ):

        csv_rows[i][
            "Node_Degree_10A"
        ] = node_degrees[i]


    # ========================================================
    # 保存feature CSV
    # ========================================================

    df_features = pd.DataFrame(
        csv_rows
    )


    cols_order = [

        "PDB_ID",
        "Chain",
        "Res_Name",
        "Res_ID",
        "Label",
        "Node_Degree_10A",

        "Sec_Struct",
        "Is_Helix",
        "Is_Sheet",
        "Is_Coil",
        "RSA",

        "Coord_X",
        "Coord_Y",
        "Coord_Z"

    ]


    df_features = (
        df_features[
            cols_order
        ]
    )


    csv_path = os.path.join(
        out_dirs[
            "csv"
        ],
        f"{pdb_id}_{chain_type}_features.csv"
    )


    df_features.to_csv(
        csv_path,
        index=False
    )


    # ========================================================
    # 保存PyG
    # ========================================================

    pyg_path = os.path.join(
        out_dirs[
            "pyg"
        ],
        f"{pdb_id}_{chain_type}.pyg"
    )


    torch.save(
        data,
        pyg_path
    )


    return True


# ============================================================
# 8. generated graph QC
# ============================================================

def qc_generated_split(
        split,
        out_dirs
):

    """
    本次只改变DSSP/RSA计算环境。
    因此graph数量、node数量和5 Å正标签数量
    必须与corrected 5A正式数据保持一致。
    """

    pyg_dir = out_dirs[
        "pyg"
    ]


    pyg_files = sorted([
        f
        for f in os.listdir(
            pyg_dir
        )
        if f.endswith(
            ".pyg"
        )
    ])


    graph_count = len(
        pyg_files
    )

    total_nodes = 0

    total_positive = 0


    for filename in pyg_files:

        graph_path = os.path.join(
            pyg_dir,
            filename
        )


        data = torch.load(
            graph_path,
            map_location="cpu",
            weights_only=False
        )


        total_nodes += int(
            data.y.numel()
        )


        total_positive += int(
            (
                data.y
                ==
                1
            ).sum().item()
        )


    expected_graphs = EXPECTED_GRAPH_COUNTS[
        split
    ]

    expected_nodes = EXPECTED_NODE_COUNTS[
        split
    ]

    expected_positive = EXPECTED_POSITIVE_COUNTS[
        split
    ]


    print(
        "\nCorrected 5A QC:"
    )

    print(
        f"  Graphs: "
        f"{graph_count} "
        f"(expected {expected_graphs})"
    )

    print(
        f"  Nodes: "
        f"{total_nodes:,} "
        f"(expected {expected_nodes:,})"
    )

    print(
        f"  Positive labels: "
        f"{total_positive:,} "
        f"(expected {expected_positive:,})"
    )


    if graph_count != expected_graphs:

        raise RuntimeError(
            f"{split}: graph count QC failed"
        )


    if total_nodes != expected_nodes:

        raise RuntimeError(
            f"{split}: node count QC failed"
        )


    if total_positive != expected_positive:

        raise RuntimeError(
            f"{split}: positive label count QC failed"
        )


    print(
        "  PASS: graph/node/label counts "
        "match corrected 5A."
    )


    return {
        "graph_count_qc":
            graph_count,

        "node_count":
            total_nodes,

        "positive_count":
            total_positive
    }


# ============================================================
# 9. 处理一个split
# ============================================================

def process_split(
        split
):

    print("\n" + "=" * 70)

    print(
        f"Processing split: {split}"
    )

    print("=" * 70)


    metadata_path = os.path.join(
        SPLIT_DIR,
        f"{split}_metadata.csv"
    )


    if not os.path.exists(
        metadata_path
    ):

        raise FileNotFoundError(
            metadata_path
        )


    df_meta = pd.read_csv(
        metadata_path
    )


    # --------------------------------------------------------
    # 输出路径
    # --------------------------------------------------------

    out_dirs = {

        "pyg":
            os.path.join(
                OUTPUT_BASE,
                f"{split}_pyg_3d_graph"
            ),

        "csv":
            os.path.join(
                OUTPUT_BASE,
                f"{split}_csv_features"
            )
    }


    for d in out_dirs.values():

        os.makedirs(
            d,
            exist_ok=True
        )


    success_complex = 0

    failed_complex = 0

    successful_graphs = 0

    failed_records = []


    # ========================================================
    # 遍历当前split metadata
    # ========================================================

    for _, row in tqdm(
        df_meta.iterrows(),
        total=len(
            df_meta
        ),
        desc=split
    ):


        pdb_file = row[
            "pdb_file"
        ]


        pdb_id = pdb_file.replace(
            ".pdb",
            ""
        )


        pdb_path = os.path.join(
            PDB_DIR,
            pdb_file
        )


        label_path = os.path.join(
            LABEL_DIR,
            split,
            f"{pdb_id}_label.csv"
        )


        # ----------------------------------------------------
        # PDB检查
        # ----------------------------------------------------

        if not os.path.exists(
            pdb_path
        ):

            print(
                f"\nMissing PDB: {pdb_file}"
            )

            failed_complex += 1

            failed_records.append({

                "pdb_file":
                    pdb_file,

                "reason":
                    "PDB file missing"
            })

            continue


        # ----------------------------------------------------
        # label检查
        # ----------------------------------------------------

        if not os.path.exists(
            label_path
        ):

            print(
                f"\nMissing label: {pdb_file}"
            )

            failed_complex += 1

            failed_records.append({

                "pdb_file":
                    pdb_file,

                "reason":
                    "label file missing"
            })

            continue


        labels_df = pd.read_csv(
            label_path
        )


        rec_chain_ids = str(
            row[
                "receptor_chain"
            ]
        )


        lig_chain_ids = str(
            row[
                "ligand_chain"
            ]
        )


        try:

            # =================================================
            # receptor独立建图
            # =================================================

            rec_ok = process_chain_graph(

                pdb_path,

                rec_chain_ids,

                labels_df,

                out_dirs,

                "rec",

                pdb_id
            )


            # =================================================
            # ligand独立建图
            # =================================================

            lig_ok = process_chain_graph(

                pdb_path,

                lig_chain_ids,

                labels_df,

                out_dirs,

                "lig",

                pdb_id
            )


            if rec_ok:
                successful_graphs += 1

            if lig_ok:
                successful_graphs += 1


            if rec_ok and lig_ok:

                success_complex += 1

            else:

                failed_complex += 1


        except Exception as e:

            failed_complex += 1

            failed_records.append({

                "pdb_file":
                    pdb_file,

                "reason":
                    str(e)
            })


            print(
                f"\nFAILED {pdb_file}: {e}"
            )


    # ========================================================
    # split统计
    # ========================================================

    print("\n" + "-" * 70)

    print(
        f"{split} finished"
    )

    print(
        f"Complex total: "
        f"{len(df_meta)}"
    )

    print(
        f"Complex success: "
        f"{success_complex}"
    )

    print(
        f"Complex failed: "
        f"{failed_complex}"
    )

    print(
        f"Graph generated: "
        f"{successful_graphs}"
    )


    # --------------------------------------------------------
    # failure log
    # --------------------------------------------------------

    failed_output = os.path.join(
        OUTPUT_BASE,
        f"{split}_failed_graphs.csv"
    )


    if failed_records:

        pd.DataFrame(
            failed_records
        ).to_csv(
            failed_output,
            index=False
        )

    else:

        pd.DataFrame(
            columns=[
                "pdb_file",
                "reason"
            ]
        ).to_csv(
            failed_output,
            index=False
        )


    # ========================================================
    # corrected 5A graph/node/positive数量QC
    # ========================================================

    qc_result = qc_generated_split(
        split,
        out_dirs
    )


    return {

        "split":
            split,

        "complex_total":
            len(df_meta),

        "complex_success":
            success_complex,

        "complex_failed":
            failed_complex,

        "graphs_generated":
            successful_graphs,

        "qc_graph_count":
            qc_result[
                "graph_count_qc"
            ],

        "qc_node_count":
            qc_result[
                "node_count"
            ],

        "qc_positive_count":
            qc_result[
                "positive_count"
            ]
    }


# ============================================================
# 10. Main
# ============================================================

def main():

    summaries = []


    for split in [
        "train",
        "val",
        "test"
    ]:

        result = process_split(
            split
        )

        summaries.append(
            result
        )


    # ========================================================
    # 总结
    # ========================================================

    summary_df = pd.DataFrame(
        summaries
    )


    summary_file = os.path.join(
        OUTPUT_BASE,
        "graph_generation_summary.csv"
    )


    summary_df.to_csv(
        summary_file,
        index=False
    )


    print("\n")
    print("=" * 70)
    print("ALL PARTNER-ONLY DSSP GRAPH CONSTRUCTION FINISHED")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nSummary saved to:"
    )

    print(
        summary_file
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()