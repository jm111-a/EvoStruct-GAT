from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Administrator\Desktop\P-P"
)


# ============================================================
# 当前正式 corrected 5A graph
#
# 不再统计旧：
# cluster_split_v3_A2/05_labels
#
# 而是统计真正进入模型的 corrected 5A graph feature CSV。
#
# 每一个 feature CSV row 与一个 graph node 对应。
# ============================================================

GRAPH_ROOT = (
    PROJECT_ROOT
    / "cluster_split_v3_A2"
    / "partner_only_DSSP"
    / "06_graphs"
)


SPLITS = {

    "train":
        GRAPH_ROOT
        / "train_csv_features",

    "val":
        GRAPH_ROOT
        / "val_csv_features",

    "test":
        GRAPH_ROOT
        / "test_csv_features",
}


# ============================================================
# 统计输出目录
# ============================================================

OUTPUT_DIR = (

    PROJECT_ROOT
    / "cluster_split_v3_A2"
    / "partner_only_DSSP"
    / "residue_statistics"

)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 当前严格划分graph数量
#
# receptor + ligand分别一个graph
# ============================================================

EXPECTED_GRAPH_COUNTS = {

    "train": 2780,

    "val": 344,

    "test": 354,

}


# ============================================================
# 当前corrected 5A预期graph-node数量
#
# 用于QC
# ============================================================

EXPECTED_NODE_COUNTS = {

    "train": 620116,

    "val": 69673,

    "test": 74086,

}


# ============================================================
# 当前corrected 5A预期positive数量
#
# 用于QC
# ============================================================

EXPECTED_POSITIVE_COUNTS = {

    "train": 69700,

    "val": 10384,

    "test": 10437,

}


# ============================================================
# 2. CSV读取
# ============================================================

def read_csv_rows(csv_path):


    encodings = [

        "utf-8-sig",

        "utf-8",

        "gbk",

    ]


    last_error = None


    for enc in encodings:


        try:


            with open(

                csv_path,

                "r",

                encoding=enc,

                newline=""

            ) as f:


                reader = csv.DictReader(
                    f
                )


                rows = list(
                    reader
                )


            return rows


        except Exception as e:


            last_error = e


    raise RuntimeError(

        f"无法读取文件: {csv_path}\n"
        f"最后错误: {last_error}"

    )


# ============================================================
# 3. 自动寻找label列
#
# corrected graph feature CSV中通常为：
#
# Label
#
# 同时兼容：
# interface_label
# ============================================================

def find_label_column(fieldnames):


    cleaned = {

        name.strip():
            name

        for name in fieldnames

    }


    # --------------------------------------------------------
    # 优先寻找 Label
    # --------------------------------------------------------

    for key in cleaned:


        if (
            key.lower()
            ==
            "label"
        ):

            return cleaned[
                key
            ]


    # --------------------------------------------------------
    # 再寻找 interface_label
    # --------------------------------------------------------

    for key in cleaned:


        if (
            key.lower()
            ==
            "interface_label"
        ):

            return cleaned[
                key
            ]


    # --------------------------------------------------------
    # 模糊匹配
    # --------------------------------------------------------

    for key in cleaned:


        low = key.lower()


        if (
            "interface" in low
            and
            "label" in low
        ):

            return cleaned[
                key
            ]


    raise ValueError(

        "没有找到Label/interface_label列，"
        f"当前列名为: {list(fieldnames)}"

    )


# ============================================================
# 4. 从文件名判断 receptor / ligand
#
# feature CSV命名：
#
# xxxx_rec_features.csv
# xxxx_lig_features.csv
# ============================================================

def determine_chain_type(
        csv_path):


    name = (
        csv_path.name
        .lower()
    )


    if name.endswith(
        "_rec_features.csv"
    ):

        return "receptor"


    elif name.endswith(
        "_lig_features.csv"
    ):

        return "ligand"


    else:

        return "unknown"


# ============================================================
# 5. 统计一个split
# ============================================================

def summarize_split(
        split_name,
        split_dir):


    """
    corrected 5A graph feature CSV：

    每个CSV对应一张receptor或ligand graph。

    CSV每一行对应一个实际graph node，
    因而这里统计的是模型真正使用的残基。

    Label:
        0 = non-interface residue
        1 = interface residue
    """


    if not split_dir.exists():


        raise FileNotFoundError(

            f"数据集目录不存在："
            f"{split_dir}"

        )


    # ========================================================
    # 当前corrected 5A feature CSV
    # ========================================================

    csv_files = sorted(

        split_dir.glob(
            "*_features.csv"
        )

    )


    print(

        f"\n[{split_name}] "
        f"找到 {len(csv_files)} 个 "
        f"corrected 5A feature CSV"

    )


    # ========================================================
    # graph数量QC
    # ========================================================

    expected_graphs = (

        EXPECTED_GRAPH_COUNTS.get(
            split_name
        )

    )


    if expected_graphs is not None:


        if (
            len(csv_files)
            !=
            expected_graphs
        ):


            print(

                f"WARNING: {split_name} "
                f"理论graph数="
                f"{expected_graphs}，"
                f"实际CSV数="
                f"{len(csv_files)}"

            )


        else:


            print(

                f"OK: graph CSV数与严格split一致 "
                f"({expected_graphs})"

            )


    # ========================================================
    # 总体统计
    # ========================================================

    total_residues = 0

    interface_residues = 0

    non_interface_residues = 0

    invalid_label_count = 0


    # ========================================================
    # receptor / ligand统计
    # ========================================================

    receptor_total = 0

    receptor_interface = 0


    ligand_total = 0

    ligand_interface = 0


    unknown_total = 0


    per_file_records = []


    # ========================================================
    # 遍历feature CSV
    # ========================================================

    for csv_file in csv_files:


        rows = read_csv_rows(
            csv_file
        )


        if not rows:

            continue


        fieldnames = (
            rows[0].keys()
        )


        label_col = (
            find_label_column(
                fieldnames
            )
        )


        chain_type = (
            determine_chain_type(
                csv_file
            )
        )


        file_total = 0

        file_interface = 0

        file_non_interface = 0

        file_invalid = 0


        # ====================================================
        # 每一行 = 一个graph node
        # ====================================================

        for row in rows:


            raw_label = str(

                row.get(
                    label_col,
                    ""
                )

            ).strip()


            try:


                label = int(

                    float(
                        raw_label
                    )

                )


            except Exception:


                file_invalid += 1

                continue


            if label not in (
                0,
                1
            ):


                file_invalid += 1

                continue


            # ------------------------------------------------
            # 总体
            # ------------------------------------------------

            file_total += 1


            if label == 1:


                file_interface += 1


            else:


                file_non_interface += 1


            # ------------------------------------------------
            # receptor / ligand
            # ------------------------------------------------

            if (
                chain_type
                ==
                "receptor"
            ):


                receptor_total += 1

                receptor_interface += label


            elif (
                chain_type
                ==
                "ligand"
            ):


                ligand_total += 1

                ligand_interface += label


            else:


                unknown_total += 1


        # ====================================================
        # 累加
        # ====================================================

        total_residues += (
            file_total
        )


        interface_residues += (
            file_interface
        )


        non_interface_residues += (
            file_non_interface
        )


        invalid_label_count += (
            file_invalid
        )


        # ====================================================
        # 单graph记录
        # ====================================================

        per_file_records.append({

            "split":
                split_name,

            "file":
                csv_file.name,

            "chain_type":
                chain_type,

            "total_residues":
                file_total,

            "interface_residues":
                file_interface,

            "non_interface_residues":
                file_non_interface,

            "interface_ratio_percent":
                (
                    file_interface
                    /
                    file_total
                    *
                    100

                    if file_total

                    else 0
                ),

            "invalid_label_count":
                file_invalid,

        })


    # ========================================================
    # 比例
    # ========================================================

    interface_ratio = (

        interface_residues
        /
        total_residues
        *
        100

        if total_residues

        else 0

    )


    neg_pos_ratio = (

        non_interface_residues
        /
        interface_residues

        if interface_residues

        else float(
            "inf"
        )

    )


    receptor_ratio = (

        receptor_interface
        /
        receptor_total
        *
        100

        if receptor_total

        else 0

    )


    ligand_ratio = (

        ligand_interface
        /
        ligand_total
        *
        100

        if ligand_total

        else 0

    )


    # ========================================================
    # Summary
    # ========================================================

    summary = {

        "split":
            split_name,

        "graph_csv_count":
            len(
                csv_files
            ),

        "total_residues":
            total_residues,

        "interface_residues":
            interface_residues,

        "non_interface_residues":
            non_interface_residues,

        "interface_ratio_percent":
            interface_ratio,

        "pos_neg_ratio":
            (
                f"1:{neg_pos_ratio:.2f}"

                if interface_residues

                else "NA"
            ),

        "invalid_label_count":
            invalid_label_count,

        "receptor_total":
            receptor_total,

        "receptor_interface":
            receptor_interface,

        "receptor_interface_ratio_percent":
            receptor_ratio,

        "ligand_total":
            ligand_total,

        "ligand_interface":
            ligand_interface,

        "ligand_interface_ratio_percent":
            ligand_ratio,

        "unknown_chain_type_nodes":
            unknown_total,

    }


    # ========================================================
    # corrected 5A强制QC
    # ========================================================

    expected_nodes = (
        EXPECTED_NODE_COUNTS.get(
            split_name
        )
    )


    expected_positive = (
        EXPECTED_POSITIVE_COUNTS.get(
            split_name
        )
    )


    if expected_nodes is not None:


        if (
            total_residues
            !=
            expected_nodes
        ):


            raise RuntimeError(

                f"{split_name}: "
                f"实际节点数="
                f"{total_residues}，"
                f"expected="
                f"{expected_nodes}"

            )


        else:


            print(

                f"OK: corrected 5A node count = "
                f"{total_residues:,}"

            )


    if expected_positive is not None:


        if (
            interface_residues
            !=
            expected_positive
        ):


            raise RuntimeError(

                f"{split_name}: "
                f"实际interface nodes="
                f"{interface_residues}，"
                f"expected="
                f"{expected_positive}"

            )


        else:


            print(

                f"OK: corrected 5A positive nodes = "
                f"{interface_residues:,}"

            )


    if invalid_label_count != 0:


        raise RuntimeError(

            f"{split_name}: "
            f"发现 {invalid_label_count} "
            f"个无效label"

        )


    if unknown_total != 0:


        raise RuntimeError(

            f"{split_name}: "
            f"发现 {unknown_total} "
            f"个无法判断rec/lig来源的nodes"

        )


    return (
        summary,
        per_file_records
    )


# ============================================================
# 6. 保存CSV
# ============================================================

def save_csv(
        path,
        records,
        fieldnames):


    with open(

        path,

        "w",

        encoding="utf-8-sig",

        newline=""

    ) as f:


        writer = csv.DictWriter(

            f,

            fieldnames=fieldnames

        )


        writer.writeheader()


        writer.writerows(
            records
        )


# ============================================================
# 7. 绘制论文图
# ============================================================

def plot_publication_residue_distribution(
        all_summaries,
        output_dir):


    """
    图形保持旧版形式：

    左轴：
        stacked bar
        non-interface + interface residues

    右轴：
        interface residue ratio

    无顶部标题
    图例位于底部
    """


    split_order = [

        "train",

        "val",

        "test",

    ]


    display_names = {

        "train":
            "Training",

        "val":
            "Validation",

        "test":
            "Test",

    }


    summaries = []


    for name in split_order:


        matched = [

            s

            for s in all_summaries

            if s[
                "split"
            ]
            ==
            name

        ]


        if matched:


            summaries.append(
                matched[0]
            )


    if not summaries:


        raise RuntimeError(
            "没有可用于绘图的数据。"
        )


    labels = [

        display_names[
            s[
                "split"
            ]
        ]

        for s in summaries

    ]


    interface_counts = np.array(

        [

            s[
                "interface_residues"
            ]

            for s in summaries

        ],

        dtype=float

    )


    non_interface_counts = np.array(

        [

            s[
                "non_interface_residues"
            ]

            for s in summaries

        ],

        dtype=float

    )


    total_counts = (

        interface_counts
        +
        non_interface_counts

    )


    interface_ratios = np.array(

        [

            s[
                "interface_ratio_percent"
            ]

            for s in summaries

        ],

        dtype=float

    )


    x = np.arange(
        len(
            labels
        )
    )


    width = 0.58


    # ========================================================
    # 字体
    # ========================================================

    plt.rcParams[
        "font.family"
    ] = "Arial"


    plt.rcParams[
        "axes.linewidth"
    ] = 1.0


    # PDF/SVG中文字可编辑

    plt.rcParams[
        "pdf.fonttype"
    ] = 42


    plt.rcParams[
        "ps.fonttype"
    ] = 42


    fig, ax1 = plt.subplots(

        figsize=(
            7.4,
            5.3
        )

    )


    # ========================================================
    # 保持原图颜色
    # ========================================================

    color_non = (
        "#BDBDBD"
    )

    color_interface = (
        "#D95F02"
    )

    color_ratio = (
        "#1B4F72"
    )


    # ========================================================
    # Non-interface bar
    # ========================================================

    ax1.bar(

        x,

        non_interface_counts,

        width=width,

        color=color_non,

        edgecolor="black",

        linewidth=0.6,

        label="Non-interface residues",

    )


    # ========================================================
    # Interface bar
    # ========================================================

    ax1.bar(

        x,

        interface_counts,

        width=width,

        bottom=non_interface_counts,

        color=color_interface,

        edgecolor="black",

        linewidth=0.6,

        label="Interface residues",

    )


    # ========================================================
    # 左轴
    # ========================================================

    ax1.set_ylabel(

        "Number of residues",

        fontsize=12

    )


    ax1.set_xticks(
        x
    )


    ax1.set_xticklabels(

        labels,

        fontsize=11

    )


    ax1.tick_params(

        axis="y",

        labelsize=10

    )


    ax1.yaxis.set_major_formatter(

        FuncFormatter(

            lambda v, _:
                f"{int(v):,}"

        )

    )


    ax1.spines[
        "top"
    ].set_visible(
        False
    )


    ax1.grid(

        axis="y",

        linestyle="--",

        linewidth=0.5,

        alpha=0.35

    )


    ax1.set_axisbelow(
        True
    )


    max_total = (

        max(
            total_counts
        )

        if len(
            total_counts
        )

        else 1

    )


    ax1.set_ylim(

        0,

        max_total
        *
        1.13

    )


    # ========================================================
    # 总残基数 n =
    # ========================================================

    for i, total in enumerate(
        total_counts
    ):


        ax1.text(

            x[i],

            total
            +
            max_total
            *
            0.018,

            f"n = {int(total):,}",

            ha="center",

            va="bottom",

            fontsize=9,

        )


    # ========================================================
    # 橙色区域标注interface数量
    # ========================================================

    for i, pos in enumerate(
        interface_counts
    ):


        if pos > 0:


            ax1.text(

                x[i],

                non_interface_counts[i]
                +
                pos
                /
                2,

                f"{int(pos):,}",

                ha="center",

                va="center",

                fontsize=9,

                color="white",

                fontweight="bold",

            )


    # ========================================================
    # 右轴
    # ========================================================

    ax2 = (
        ax1.twinx()
    )


    ax2.plot(

        x,

        interface_ratios,

        marker="o",

        markersize=6,

        linewidth=1.8,

        color=color_ratio,

        label="Interface residue ratio",

    )


    ax2.set_ylabel(

        "Interface residue ratio (%)",

        fontsize=12

    )


    ax2.tick_params(

        axis="y",

        labelsize=10

    )


    ax2.spines[
        "top"
    ].set_visible(
        False
    )


    max_ratio = (

        max(
            interface_ratios
        )

        if len(
            interface_ratios
        )

        else 1

    )


    ax2.set_ylim(

        0,

        max_ratio
        *
        1.35

    )


    # ========================================================
    # 比例标注
    # ========================================================

    for i, ratio in enumerate(
        interface_ratios
    ):


        ax2.text(

            x[i],

            ratio
            +
            max_ratio
            *
            0.045,

            f"{ratio:.2f}%",

            ha="center",

            va="bottom",

            fontsize=9,

            color=color_ratio,

            fontweight="bold",

        )


    # ========================================================
    # 图例
    # ========================================================

    handles1, labels1 = (
        ax1.get_legend_handles_labels()
    )


    handles2, labels2 = (
        ax2.get_legend_handles_labels()
    )


    ax1.legend(

        handles1
        +
        handles2,

        labels1
        +
        labels2,

        loc="upper center",

        bbox_to_anchor=(
            0.5,
            -0.12
        ),

        ncol=3,

        frameon=False,

        fontsize=10,

    )


    # ========================================================
    # 不加顶部标题
    # ========================================================

    plt.tight_layout(

        rect=[
            0,
            0.08,
            1,
            1
        ]

    )


    # ========================================================
    # 输出
    # ========================================================

    png_path = (

        output_dir
        /
        "Figure_dataset_residue_distribution_corrected5A.png"

    )


    jpg_path = (

        output_dir
        /
        "Figure_dataset_residue_distribution_corrected5A.jpg"

    )


    tiff_path = (

        output_dir
        /
        "Figure_dataset_residue_distribution_corrected5A.tiff"

    )


    pdf_path = (

        output_dir
        /
        "Figure_dataset_residue_distribution_corrected5A.pdf"

    )


    svg_path = (

        output_dir
        /
        "Figure_dataset_residue_distribution_corrected5A.svg"

    )


    fig.savefig(

        png_path,

        dpi=600,

        bbox_inches="tight"

    )


    fig.savefig(

        jpg_path,

        dpi=600,

        bbox_inches="tight"

    )


    fig.savefig(

        tiff_path,

        dpi=600,

        bbox_inches="tight"

    )


    fig.savefig(

        pdf_path,

        bbox_inches="tight"

    )


    fig.savefig(

        svg_path,

        bbox_inches="tight"

    )


    plt.close(
        fig
    )


    print(
        "\n已保存corrected 5A数据集分布图："
    )


    print(
        png_path
    )


    print(
        jpg_path
    )


    print(
        tiff_path
    )


    print(
        pdf_path
    )


    print(
        svg_path
    )


# ============================================================
# 8. 主程序
# ============================================================

def main():


    all_summaries = []

    all_per_file_records = []


    print(
        "=" * 74
    )


    print(
        "Corrected 5A graph-node residue statistics"
    )


    print(
        "Strict cluster-disjoint dataset"
    )


    print(
        "=" * 74
    )


    print(
        "\nGraph root:"
    )


    print(
        GRAPH_ROOT
    )


    # ========================================================
    # 统计三个split
    # ========================================================

    for (
        split_name,
        split_dir
    ) in SPLITS.items():


        summary, per_file_records = (

            summarize_split(

                split_name,

                split_dir

            )

        )


        all_summaries.append(
            summary
        )


        all_per_file_records.extend(
            per_file_records
        )


    # ========================================================
    # 打印结果
    # ========================================================

    print(

        "\n========== "
        "Corrected 5A 数据集残基统计 "
        "==========\n"

    )


    for s in all_summaries:


        print(

            f"数据集: "
            f"{s['split']}"

        )


        print(

            f"  Graph/CSV数: "
            f"{s['graph_csv_count']}"

        )


        print(

            f"  总graph nodes/残基数: "
            f"{s['total_residues']:,}"

        )


        print(

            f"  界面残基数: "
            f"{s['interface_residues']:,}"

        )


        print(

            f"  非界面残基数: "
            f"{s['non_interface_residues']:,}"

        )


        print(

            f"  界面残基比例: "
            f"{s['interface_ratio_percent']:.2f}%"

        )


        print(

            f"  正负样本比例: "
            f"{s['pos_neg_ratio']}"

        )


        print(

            f"  无效标签数: "
            f"{s['invalid_label_count']}"

        )


        # ====================================================
        # receptor
        # ====================================================

        if (
            s[
                "receptor_total"
            ]
            >
            0
        ):


            print(

                f"  Receptor残基: "
                f"{s['receptor_total']:,}"

            )


            print(

                f"  Receptor界面残基: "
                f"{s['receptor_interface']:,}"

            )


            print(

                f"  Receptor界面比例: "
                f"{s['receptor_interface_ratio_percent']:.2f}%"

            )


        # ====================================================
        # ligand
        # ====================================================

        if (
            s[
                "ligand_total"
            ]
            >
            0
        ):


            print(

                f"  Ligand残基: "
                f"{s['ligand_total']:,}"

            )


            print(

                f"  Ligand界面残基: "
                f"{s['ligand_interface']:,}"

            )


            print(

                f"  Ligand界面比例: "
                f"{s['ligand_interface_ratio_percent']:.2f}%"

            )


        print()


    # ========================================================
    # 保存summary
    # ========================================================

    summary_path = (

        OUTPUT_DIR
        /
        "corrected5A_residue_distribution_summary.csv"

    )


    per_file_path = (

        OUTPUT_DIR
        /
        "corrected5A_residue_distribution_per_graph.csv"

    )


    summary_fields = [

        "split",

        "graph_csv_count",

        "total_residues",

        "interface_residues",

        "non_interface_residues",

        "interface_ratio_percent",

        "pos_neg_ratio",

        "invalid_label_count",

        "receptor_total",

        "receptor_interface",

        "receptor_interface_ratio_percent",

        "ligand_total",

        "ligand_interface",

        "ligand_interface_ratio_percent",

        "unknown_chain_type_nodes",

    ]


    per_file_fields = [

        "split",

        "file",

        "chain_type",

        "total_residues",

        "interface_residues",

        "non_interface_residues",

        "interface_ratio_percent",

        "invalid_label_count",

    ]


    save_csv(

        summary_path,

        all_summaries,

        summary_fields

    )


    save_csv(

        per_file_path,

        all_per_file_records,

        per_file_fields

    )


    print(
        "已保存统计表："
    )


    print(
        summary_path
    )


    print(
        per_file_path
    )


    # ========================================================
    # 绘图
    # ========================================================

    plot_publication_residue_distribution(

        all_summaries,

        OUTPUT_DIR

    )


    # ========================================================
    # 最终QC
    # ========================================================

    print(
        "\n"
        +
        "=" * 74
    )


    print(
        "FINAL CORRECTED 5A QC"
    )


    print(
        "=" * 74
    )


    for s in all_summaries:


        print(

            f"{s['split']}: "
            f"nodes={s['total_residues']:,}, "
            f"positive={s['interface_residues']:,}, "
            f"ratio="
            f"{s['interface_ratio_percent']:.4f}%"

        )


    print(
        "\n✅ Corrected 5A residue statistics completed."
    )


    print(
        "=" * 74
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()