import os
import copy
import torch
import pandas as pd


# ============================================================
# 0. 基础配置
# ============================================================

ROOT = r"C:\Users\Administrator\Desktop\P-P"

BASE_DIR = os.path.join(
    ROOT,
    "cluster_split_v3_A2"
)

# 新版正式 5 Å partner-only DSSP graph
# sensitivity analysis 只复用这里的 x / edge_index / edge_attr / pos
SOURCE_GRAPH_ROOT = os.path.join(
    BASE_DIR,
    "partner_only_DSSP",
    "06_graphs"
)

# sensitivity 根目录
SENSITIVITY_ROOT = os.path.join(
    BASE_DIR,
    "sensitivity_analysis"
)

EXPERIMENTS = [
    "4A",
    "4p5A",
    "deltaSASA1",
]

SPLITS = [
    "train",
    "val",
    "test",
]


# ============================================================
# 1. QC 预期值
#
# graph/node 数应与正式 partner-only 5 Å graph 完全一致；
# positive 数来自已经校正过的 4A / 4.5A / ΔSASA 标签。
# ============================================================

EXPECTED = {

    "4A": {
        "train": {
            "graphs": 2780,
            "nodes": 620116,
            "positive": 54954,
        },
        "val": {
            "graphs": 344,
            "nodes": 69673,
            "positive": 8301,
        },
        "test": {
            "graphs": 354,
            "nodes": 74086,
            "positive": 8284,
        },
    },

    "4p5A": {
        "train": {
            "graphs": 2780,
            "nodes": 620116,
            "positive": 63060,
        },
        "val": {
            "graphs": 344,
            "nodes": 69673,
            "positive": 9484,
        },
        "test": {
            "graphs": 354,
            "nodes": 74086,
            "positive": 9502,
        },
    },

    "deltaSASA1": {
        "train": {
            "graphs": 2780,
            "nodes": 620116,
            "positive": 75849,
        },
        "val": {
            "graphs": 344,
            "nodes": 69673,
            "positive": 11149,
        },
        "test": {
            "graphs": 354,
            "nodes": 74086,
            "positive": 11487,
        },
    },
}


# ============================================================
# 2. 通用工具
# ============================================================

def load_pyg(path):

    try:

        return torch.load(
            path,
            weights_only=False
        )

    except Exception:

        return torch.load(
            path
        )


def normalize_residue_index(value):

    try:

        return int(value)

    except Exception:

        return value


def get_pdb_and_chain_type(graph_filename):

    if graph_filename.endswith("_rec.pyg"):

        return (
            graph_filename[:-len("_rec.pyg")],
            "rec"
        )

    if graph_filename.endswith("_lig.pyg"):

        return (
            graph_filename[:-len("_lig.pyg")],
            "lig"
        )

    raise ValueError(
        f"无法识别 graph 文件名: {graph_filename}"
    )


def build_new_y(
        feature_csv,
        label_csv
):

    """
    严格按 graph node / feature CSV 的顺序替换标签。

    每个节点逐一核对：
        Chain
        Res_ID
        Res_Name

    这样保留 corrected 5 Å 重建时采用的安全映射逻辑，
    避免旧版仅按 chain_id + residue_index 查找时可能出现的
    insertion-code / duplicate residue index 问题。
    """

    feature_df = pd.read_csv(
        feature_csv
    )

    label_df = pd.read_csv(
        label_csv
    )


    required_feature_cols = {
        "Chain",
        "Res_ID",
        "Res_Name",
    }

    required_label_cols = {
        "chain_id",
        "residue_index",
        "residue_name",
        "interface_label",
    }


    missing_feature = (
        required_feature_cols
        -
        set(feature_df.columns)
    )

    missing_label = (
        required_label_cols
        -
        set(label_df.columns)
    )


    if missing_feature:

        raise ValueError(
            f"{feature_csv} 缺少列: "
            f"{sorted(missing_feature)}"
        )


    if missing_label:

        raise ValueError(
            f"{label_csv} 缺少列: "
            f"{sorted(missing_label)}"
        )


    target_chain_ids = (
        feature_df["Chain"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )


    relevant_labels = (
        label_df[
            label_df["chain_id"]
            .astype(str)
            .isin(target_chain_ids)
        ]
        .copy()
        .reset_index(drop=True)
    )


    # 如果 target partner >1022 aa，graph 可能只保留前1022个节点。
    # 因此允许 label rows 多于 feature rows，但不能少于。
    if len(relevant_labels) < len(feature_df):

        raise ValueError(
            f"label rows不足: "
            f"feature={len(feature_df)}, "
            f"labels={len(relevant_labels)}"
        )


    relevant_labels = (
        relevant_labels
        .iloc[:len(feature_df)]
        .reset_index(drop=True)
    )


    new_y = []


    for i in range(len(feature_df)):

        f_row = feature_df.iloc[i]
        l_row = relevant_labels.iloc[i]


        f_chain = str(
            f_row["Chain"]
        )

        f_res_id = normalize_residue_index(
            f_row["Res_ID"]
        )

        f_res_name = str(
            f_row["Res_Name"]
        ).upper()


        l_chain = str(
            l_row["chain_id"]
        )

        l_res_id = normalize_residue_index(
            l_row["residue_index"]
        )

        l_res_name = str(
            l_row["residue_name"]
        ).upper()


        if (
            f_chain != l_chain
            or
            f_res_id != l_res_id
            or
            f_res_name != l_res_name
        ):

            raise ValueError(
                "label-order mismatch "
                f"at node {i}: "
                f"Feature=({f_chain},{f_res_id},{f_res_name}), "
                f"Label=({l_chain},{l_res_id},{l_res_name})"
            )


        new_y.append(
            int(
                l_row["interface_label"]
            )
        )


    return (
        feature_df,
        torch.tensor(
            new_y,
            dtype=torch.long
        )
    )


def prepare_output_dirs(
        experiment,
        split
):

    graph_root = os.path.join(
        SENSITIVITY_ROOT,
        experiment,
        "06_graphs"
    )


    pyg_dir = os.path.join(
        graph_root,
        f"{split}_pyg_3d_graph"
    )

    csv_dir = os.path.join(
        graph_root,
        f"{split}_csv_features"
    )


    os.makedirs(
        pyg_dir,
        exist_ok=True
    )

    os.makedirs(
        csv_dir,
        exist_ok=True
    )


    return (
        pyg_dir,
        csv_dir
    )


# ============================================================
# 3. 单个 experiment / split
# ============================================================

def process_split(
        experiment,
        split
):

    source_pyg_dir = os.path.join(
        SOURCE_GRAPH_ROOT,
        f"{split}_pyg_3d_graph"
    )

    source_csv_dir = os.path.join(
        SOURCE_GRAPH_ROOT,
        f"{split}_csv_features"
    )

    label_dir = os.path.join(
        SENSITIVITY_ROOT,
        experiment,
        "05_labels",
        split
    )


    if not os.path.isdir(source_pyg_dir):

        raise FileNotFoundError(
            f"找不到 source pyg dir:\n"
            f"{source_pyg_dir}"
        )


    if not os.path.isdir(source_csv_dir):

        raise FileNotFoundError(
            f"找不到 source feature csv dir:\n"
            f"{source_csv_dir}"
        )


    if not os.path.isdir(label_dir):

        raise FileNotFoundError(
            f"找不到 {experiment} label dir:\n"
            f"{label_dir}"
        )


    out_pyg_dir, out_csv_dir = (
        prepare_output_dirs(
            experiment,
            split
        )
    )


    graph_files = sorted([
        f
        for f in os.listdir(source_pyg_dir)
        if f.endswith(".pyg")
    ])


    print(
        "\n"
        +
        "=" * 72
    )

    print(
        f"{experiment} | {split}"
    )

    print(
        f"Source graphs: {len(graph_files)}"
    )

    print(
        "=" * 72
    )


    total_nodes = 0
    total_positive = 0
    success = 0
    failed = []


    for idx, graph_file in enumerate(
        graph_files,
        start=1
    ):

        try:

            pdb_id, chain_type = (
                get_pdb_and_chain_type(
                    graph_file
                )
            )


            source_pyg = os.path.join(
                source_pyg_dir,
                graph_file
            )


            feature_csv = os.path.join(
                source_csv_dir,
                f"{pdb_id}_{chain_type}_features.csv"
            )


            label_csv = os.path.join(
                label_dir,
                f"{pdb_id}_label.csv"
            )


            if not os.path.exists(feature_csv):

                raise FileNotFoundError(
                    f"缺少 feature CSV: "
                    f"{feature_csv}"
                )


            if not os.path.exists(label_csv):

                raise FileNotFoundError(
                    f"缺少 label CSV: "
                    f"{label_csv}"
                )


            source_data = load_pyg(
                source_pyg
            )


            feature_df, new_y = (
                build_new_y(
                    feature_csv,
                    label_csv
                )
            )


            if source_data.x.shape[0] != len(new_y):

                raise ValueError(
                    f"node/y数量不一致: "
                    f"x={source_data.x.shape[0]}, "
                    f"y={len(new_y)}"
                )


            if len(feature_df) != source_data.x.shape[0]:

                raise ValueError(
                    f"feature/node数量不一致: "
                    f"feature={len(feature_df)}, "
                    f"node={source_data.x.shape[0]}"
                )


            # -----------------------------------------------
            # 只替换 y
            # x / edge_index / edge_attr / pos 保持原样
            # -----------------------------------------------

            new_data = copy.deepcopy(
                source_data
            )


            new_data.y = (
                new_y.to(
                    dtype=source_data.y.dtype
                )
            )


            # 核心 graph 内容严格检查
            if not torch.equal(
                new_data.x,
                source_data.x
            ):

                raise RuntimeError(
                    "x发生变化"
                )


            if not torch.equal(
                new_data.edge_index,
                source_data.edge_index
            ):

                raise RuntimeError(
                    "edge_index发生变化"
                )


            if source_data.edge_attr is not None:

                if not torch.equal(
                    new_data.edge_attr,
                    source_data.edge_attr
                ):

                    raise RuntimeError(
                        "edge_attr发生变化"
                    )


            if (
                hasattr(source_data, "pos")
                and
                source_data.pos is not None
            ):

                if not torch.equal(
                    new_data.pos,
                    source_data.pos
                ):

                    raise RuntimeError(
                        "pos发生变化"
                    )


            out_pyg = os.path.join(
                out_pyg_dir,
                graph_file
            )


            torch.save(
                new_data,
                out_pyg
            )


            # 同步更新 feature CSV 中 Label 列
            # 其余特征全部保持不变
            out_feature_df = (
                feature_df.copy()
            )


            out_feature_df["Label"] = (
                new_y.cpu().numpy()
            )


            out_feature_csv = os.path.join(
                out_csv_dir,
                f"{pdb_id}_{chain_type}_features.csv"
            )


            out_feature_df.to_csv(
                out_feature_csv,
                index=False
            )


            total_nodes += int(
                new_data.y.numel()
            )

            total_positive += int(
                new_data.y.sum().item()
            )

            success += 1


            if (
                idx % 200 == 0
                or
                idx == len(graph_files)
            ):

                print(
                    f"[{idx}/{len(graph_files)}] "
                    f"success={success}, "
                    f"failed={len(failed)}"
                )


        except Exception as e:

            failed.append({
                "graph_file": graph_file,
                "reason": str(e),
            })

            print(
                f"FAILED: "
                f"{graph_file} | "
                f"{e}"
            )


    # ========================================================
    # split 级 QC
    # ========================================================

    expected = EXPECTED[
        experiment
    ][
        split
    ]


    print(
        "\nQC:"
    )

    print(
        "Graphs:",
        success,
        "| expected:",
        expected["graphs"]
    )

    print(
        "Nodes:",
        total_nodes,
        "| expected:",
        expected["nodes"]
    )

    print(
        "Positive:",
        total_positive,
        "| expected:",
        expected["positive"]
    )

    print(
        "Failed:",
        len(failed)
    )


    if success != expected["graphs"]:

        raise RuntimeError(
            f"{experiment}/{split}: "
            "graph count QC failed"
        )


    if total_nodes != expected["nodes"]:

        raise RuntimeError(
            f"{experiment}/{split}: "
            "node count QC failed"
        )


    if total_positive != expected["positive"]:

        raise RuntimeError(
            f"{experiment}/{split}: "
            "positive count QC failed"
        )


    if failed:

        failed_csv = os.path.join(
            SENSITIVITY_ROOT,
            experiment,
            "06_graphs",
            f"{split}_failed_graphs.csv"
        )

        pd.DataFrame(
            failed
        ).to_csv(
            failed_csv,
            index=False,
            encoding="utf-8-sig"
        )

        raise RuntimeError(
            f"{experiment}/{split}: "
            f"存在 {len(failed)} 个失败 graph"
        )


    return {
        "experiment": experiment,
        "split": split,
        "graphs": success,
        "nodes": total_nodes,
        "positive": total_positive,
        "failed": len(failed),
    }


# ============================================================
# 4. Main
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "Partner-only DSSP sensitivity graph generation"
    )

    print(
        "Source graph:"
    )

    print(
        SOURCE_GRAPH_ROOT
    )

    print(
        "\nOutput:"
    )

    print(
        SENSITIVITY_ROOT
    )

    print(
        "\nOnly y labels will be replaced."
    )

    print(
        "x / edge_index / edge_attr / pos remain unchanged."
    )

    print(
        "=" * 72
    )


    if not os.path.isdir(SOURCE_GRAPH_ROOT):

        raise FileNotFoundError(
            f"找不到新版 partner-only graph root:\n"
            f"{SOURCE_GRAPH_ROOT}"
        )


    # 检查三套标签目录
    for experiment in EXPERIMENTS:

        for split in SPLITS:

            label_dir = os.path.join(
                SENSITIVITY_ROOT,
                experiment,
                "05_labels",
                split
            )

            if not os.path.isdir(label_dir):

                raise FileNotFoundError(
                    f"找不到 label dir:\n"
                    f"{label_dir}"
                )


    summary_rows = []


    for experiment in EXPERIMENTS:

        for split in SPLITS:

            result = process_split(
                experiment,
                split
            )

            summary_rows.append(
                result
            )


    summary_csv = os.path.join(
        SENSITIVITY_ROOT,
        "sensitivity_graph_generation_summary.csv"
    )


    pd.DataFrame(
        summary_rows
    ).to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "ALL SENSITIVITY GRAPHS PASSED QC"
    )

    print(
        "Summary:"
    )

    print(
        summary_csv
    )

    print(
        "=" * 72
    )
