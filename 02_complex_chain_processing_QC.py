import os
import pandas as pd

from Bio.PDB import PDBParser


# ============================================================
# 1. 路径配置
# ============================================================

ROOT = r"C:\Users\Administrator\Desktop\P-P"


# ------------------------------------------------------------
# 当前正式使用的 chain-trimmed PDB
# ------------------------------------------------------------

PDB_DIR = os.path.join(
    ROOT,
    "03_chain_trimmed_pdb"
)


# ------------------------------------------------------------
# 当前正式 strict cluster-disjoint split
#
# train = 1390
# val   = 172
# test  = 177
# ------------------------------------------------------------

SPLIT_DIR = os.path.join(
    ROOT,
    "cluster_split_v3_A2",
    "04_split"
)


# ------------------------------------------------------------
# QC输出目录
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    ROOT,
    "cluster_split_v3_A2",
    "chain_processing_QC"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. 输出文件
# ============================================================

QC_CSV = os.path.join(
    OUTPUT_DIR,
    "complex_chain_processing_qc.csv"
)


SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "complex_chain_processing_summary.csv"
)


FAILURE_CSV = os.path.join(
    OUTPUT_DIR,
    "complex_chain_processing_failures.csv"
)


MULTICHAIN_CSV = os.path.join(
    OUTPUT_DIR,
    "multichain_complexes.csv"
)


# ============================================================
# 3. 标准氨基酸
# ============================================================

STANDARD_AA = {

    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR"

}


# ============================================================
# 4. PDB parser
# ============================================================

parser = PDBParser(
    QUIET=True
)


# ============================================================
# 5. chain ID解析
#
# 兼容：
#
# A
# HL
# AB
# A,B
# A B
# A;B
#
# PDB chain ID本身通常为单字符。
# ============================================================

def parse_chain_ids(
        value):


    if pd.isna(
        value
    ):

        return []


    s = str(
        value
    )


    s = (
        s
        .replace(",", "")
        .replace(" ", "")
        .replace(";", "")
        .strip()
    )


    if s == "":

        return []


    return list(
        s
    )


# ============================================================
# 6. 统一chain list显示格式
# ============================================================

def format_chain_list(
        chains):


    if not chains:

        return ""


    return ",".join(
        chains
    )


# ============================================================
# 7. 获取一个chain中的标准AA数
# ============================================================

def count_standard_residues(
        chain):


    count = 0


    for residue in chain:


        if (
            residue.get_id()[0] == " "
            and
            residue.get_resname() in STANDARD_AA
        ):

            count += 1


    return count


# ============================================================
# 8. 检查PDB中非标准/hetero residues
#
# 理论上03_chain_trimmed_pdb来源于：
#
# protein-only clean PDB
#
# 因此这里应当全部为0。
# ============================================================

def find_nonstandard_residues(
        model):


    records = []


    for chain in model:


        for residue in chain:


            is_standard = (

                residue.get_id()[0] == " "
                and
                residue.get_resname() in STANDARD_AA

            )


            if not is_standard:


                records.append(

                    f"{chain.id}:"
                    f"{residue.get_resname()}:"
                    f"{residue.get_id()}"

                )


    return records


# ============================================================
# 9. 检查单个complex
# ============================================================

def check_one_complex(
        row,
        split):


    pdb_file = str(
        row[
            "pdb_file"
        ]
    )


    pdb_id = pdb_file.replace(
        ".pdb",
        ""
    )


    # ========================================================
    # metadata receptor / ligand
    # ========================================================

    receptor_chains = parse_chain_ids(
        row[
            "receptor_chain"
        ]
    )


    ligand_chains = parse_chain_ids(
        row[
            "ligand_chain"
        ]
    )


    # --------------------------------------------------------
    # 保留metadata原始表示
    # --------------------------------------------------------

    receptor_chain_raw = str(
        row[
            "receptor_chain"
        ]
    )


    ligand_chain_raw = str(
        row[
            "ligand_chain"
        ]
    )


    expected_chains = sorted(
        set(
            receptor_chains
            +
            ligand_chains
        )
    )


    # ========================================================
    # receptor / ligand chain overlap
    # ========================================================

    receptor_ligand_overlap = sorted(

        set(
            receptor_chains
        )

        &

        set(
            ligand_chains
        )

    )


    # ========================================================
    # PDB path
    # ========================================================

    pdb_path = os.path.join(
        PDB_DIR,
        pdb_file
    )


    # ========================================================
    # 默认结果
    # ========================================================

    result = {

        "split":
            split,

        "pdb_id":
            pdb_id,

        "pdb_file":
            pdb_file,

        "receptor_chain_metadata":
            receptor_chain_raw,

        "ligand_chain_metadata":
            ligand_chain_raw,

        "receptor_chains":
            format_chain_list(
                receptor_chains
            ),

        "ligand_chains":
            format_chain_list(
                ligand_chains
            ),

        "expected_chains":
            format_chain_list(
                expected_chains
            ),

        "n_receptor_chains":
            len(
                receptor_chains
            ),

        "n_ligand_chains":
            len(
                ligand_chains
            ),

        "n_expected_chains":
            len(
                expected_chains
            ),

        "actual_chains":
            "",

        "n_actual_chains":
            0,

        "missing_receptor_chains":
            "",

        "missing_ligand_chains":
            "",

        "missing_expected_chains":
            "",

        "unexpected_extra_chains":
            "",

        "receptor_ligand_overlap":
            format_chain_list(
                receptor_ligand_overlap
            ),

        "receptor_residue_count":
            0,

        "ligand_residue_count":
            0,

        "total_standard_residue_count":
            0,

        "nonstandard_or_hetero_residue_count":
            0,

        "nonstandard_examples":
            "",

        "complex_type":
            "",

        "exact_chain_match":
            False,

        "status":
            "FAIL",

        "failure_reason":
            ""

    }


    # ========================================================
    # complex类别
    # ========================================================

    n_rec = len(
        receptor_chains
    )


    n_lig = len(
        ligand_chains
    )


    if (
        n_rec == 1
        and
        n_lig == 1
    ):

        complex_type = (
            "single_receptor_single_ligand"
        )


    elif (
        n_rec > 1
        and
        n_lig == 1
    ):

        complex_type = (
            "multichain_receptor_single_ligand"
        )


    elif (
        n_rec == 1
        and
        n_lig > 1
    ):

        complex_type = (
            "single_receptor_multichain_ligand"
        )


    elif (
        n_rec > 1
        and
        n_lig > 1
    ):

        complex_type = (
            "multichain_receptor_multichain_ligand"
        )


    else:

        complex_type = (
            "invalid_chain_assignment"
        )


    result[
        "complex_type"
    ] = complex_type


    # ========================================================
    # Failure reasons
    # ========================================================

    failure_reasons = []


    # --------------------------------------------------------
    # metadata本身检查
    # --------------------------------------------------------

    if len(
        receptor_chains
    ) == 0:

        failure_reasons.append(
            "No receptor chain in metadata"
        )


    if len(
        ligand_chains
    ) == 0:

        failure_reasons.append(
            "No ligand chain in metadata"
        )


    if receptor_ligand_overlap:

        failure_reasons.append(

            "Receptor/ligand chain overlap: "
            +
            format_chain_list(
                receptor_ligand_overlap
            )

        )


    # --------------------------------------------------------
    # PDB存在性
    # --------------------------------------------------------

    if not os.path.exists(
        pdb_path
    ):


        failure_reasons.append(
            "Trimmed PDB missing"
        )


        result[
            "failure_reason"
        ] = "; ".join(
            failure_reasons
        )


        return result


    # ========================================================
    # 解析PDB
    # ========================================================

    try:


        structure = parser.get_structure(
            pdb_id,
            pdb_path
        )


        model = structure[
            0
        ]


    except Exception as e:


        failure_reasons.append(

            "PDB parse failed: "
            +
            str(
                e
            )

        )


        result[
            "failure_reason"
        ] = "; ".join(
            failure_reasons
        )


        return result


    # ========================================================
    # 实际chains
    #
    # 只统计至少包含一个标准AA residue的chains
    # ========================================================

    actual_chains = []


    chain_residue_counts = {}


    for chain in model:


        aa_count = count_standard_residues(
            chain
        )


        if aa_count > 0:


            actual_chains.append(
                chain.id
            )


            chain_residue_counts[
                chain.id
            ] = aa_count


    actual_chains = sorted(
        actual_chains
    )


    result[
        "actual_chains"
    ] = format_chain_list(
        actual_chains
    )


    result[
        "n_actual_chains"
    ] = len(
        actual_chains
    )


    # ========================================================
    # missing chains
    # ========================================================

    missing_receptor = sorted([

        c

        for c in receptor_chains

        if c not in actual_chains

    ])


    missing_ligand = sorted([

        c

        for c in ligand_chains

        if c not in actual_chains

    ])


    missing_expected = sorted([

        c

        for c in expected_chains

        if c not in actual_chains

    ])


    result[
        "missing_receptor_chains"
    ] = format_chain_list(
        missing_receptor
    )


    result[
        "missing_ligand_chains"
    ] = format_chain_list(
        missing_ligand
    )


    result[
        "missing_expected_chains"
    ] = format_chain_list(
        missing_expected
    )


    # ========================================================
    # extra chains
    # ========================================================

    extra_chains = sorted([

        c

        for c in actual_chains

        if c not in expected_chains

    ])


    result[
        "unexpected_extra_chains"
    ] = format_chain_list(
        extra_chains
    )


    # ========================================================
    # exact chain match
    # ========================================================

    exact_chain_match = (

        set(
            actual_chains
        )

        ==

        set(
            expected_chains
        )

    )


    result[
        "exact_chain_match"
    ] = exact_chain_match


    # ========================================================
    # residue counts
    # ========================================================

    receptor_residue_count = sum(

        chain_residue_counts.get(
            c,
            0
        )

        for c in receptor_chains

    )


    ligand_residue_count = sum(

        chain_residue_counts.get(
            c,
            0
        )

        for c in ligand_chains

    )


    total_standard_residue_count = sum(
        chain_residue_counts.values()
    )


    result[
        "receptor_residue_count"
    ] = receptor_residue_count


    result[
        "ligand_residue_count"
    ] = ligand_residue_count


    result[
        "total_standard_residue_count"
    ] = total_standard_residue_count


    # ========================================================
    # 非标准 / hetero residues
    # ========================================================

    nonstandard_records = (
        find_nonstandard_residues(
            model
        )
    )


    result[
        "nonstandard_or_hetero_residue_count"
    ] = len(
        nonstandard_records
    )


    if nonstandard_records:


        result[
            "nonstandard_examples"
        ] = " | ".join(
            nonstandard_records[
                :10
            ]
        )


    # ========================================================
    # Failure conditions
    # ========================================================

    if missing_receptor:


        failure_reasons.append(

            "Missing receptor chain(s): "
            +
            format_chain_list(
                missing_receptor
            )

        )


    if missing_ligand:


        failure_reasons.append(

            "Missing ligand chain(s): "
            +
            format_chain_list(
                missing_ligand
            )

        )


    if extra_chains:


        failure_reasons.append(

            "Unexpected extra chain(s): "
            +
            format_chain_list(
                extra_chains
            )

        )


    if (
        receptor_residue_count
        == 0
    ):

        failure_reasons.append(
            "Receptor has zero standard residues"
        )


    if (
        ligand_residue_count
        == 0
    ):

        failure_reasons.append(
            "Ligand has zero standard residues"
        )


    if nonstandard_records:


        failure_reasons.append(

            "Nonstandard/hetero residues remain: "
            +
            str(
                len(
                    nonstandard_records
                )
            )

        )


    # ========================================================
    # Status
    # ========================================================

    if len(
        failure_reasons
    ) == 0:


        result[
            "status"
        ] = "PASS"


        result[
            "failure_reason"
        ] = ""


    else:


        result[
            "status"
        ] = "FAIL"


        result[
            "failure_reason"
        ] = "; ".join(
            failure_reasons
        )


    return result


# ============================================================
# 10. 主流程
# ============================================================

def main():


    print(
        "=" * 78
    )


    print(
        "Complex Chain Processing QC"
    )


    print(
        "Strict cluster-disjoint dataset"
    )


    print(
        "=" * 78
    )


    print(
        "\nPDB directory:"
    )


    print(
        PDB_DIR
    )


    print(
        "\nSplit directory:"
    )


    print(
        SPLIT_DIR
    )


    all_results = []


    # ========================================================
    # split逐个处理
    # ========================================================

    for split in [

        "train",
        "val",
        "test"

    ]:


        print(
            "\n"
            +
            "=" * 78
        )


        print(
            f"Split: {split}"
        )


        print(
            "=" * 78
        )


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


        df = pd.read_csv(
            metadata_path
        )


        print(
            "Complexes:",
            len(
                df
            )
        )


        split_results = []


        for _, row in df.iterrows():


            result = check_one_complex(

                row,

                split

            )


            split_results.append(
                result
            )


        split_df = pd.DataFrame(
            split_results
        )


        all_results.extend(
            split_results
        )


        pass_count = int(
            (
                split_df[
                    "status"
                ]
                ==
                "PASS"
            ).sum()
        )


        fail_count = int(
            (
                split_df[
                    "status"
                ]
                ==
                "FAIL"
            ).sum()
        )


        print(
            "\n---------- Split Summary ----------"
        )


        print(
            "Total:",
            len(
                split_df
            )
        )


        print(
            "PASS:",
            pass_count
        )


        print(
            "FAIL:",
            fail_count
        )


        print(
            "Exact chain match:",
            int(
                split_df[
                    "exact_chain_match"
                ].sum()
            )
        )


        print(
            "Receptor/ligand overlap:",
            int(
                (
                    split_df[
                        "receptor_ligand_overlap"
                    ]
                    !=
                    ""
                ).sum()
            )
        )


        print(
            "Unexpected extra chains:",
            int(
                (
                    split_df[
                        "unexpected_extra_chains"
                    ]
                    !=
                    ""
                ).sum()
            )
        )


        print(
            "Nonstandard/hetero residue complexes:",
            int(
                (
                    split_df[
                        "nonstandard_or_hetero_residue_count"
                    ]
                    >
                    0
                ).sum()
            )
        )


    # ========================================================
    # 11. 总QC表
    # ========================================================

    qc_df = pd.DataFrame(
        all_results
    )


    qc_df.to_csv(

        QC_CSV,

        index=False,

        encoding="utf-8-sig"

    )


    # ========================================================
    # 12. Failure table
    # ========================================================

    failure_df = qc_df[

        qc_df[
            "status"
        ]
        ==
        "FAIL"

    ].copy()


    failure_df.to_csv(

        FAILURE_CSV,

        index=False,

        encoding="utf-8-sig"

    )


    # ========================================================
    # 13. Multichain table
    #
    # 只要任意一侧 >1 chain
    # 就归入multichain
    # ========================================================

    multichain_df = qc_df[

        (
            qc_df[
                "n_receptor_chains"
            ]
            >
            1
        )

        |

        (
            qc_df[
                "n_ligand_chains"
            ]
            >
            1
        )

    ].copy()


    multichain_df.to_csv(

        MULTICHAIN_CSV,

        index=False,

        encoding="utf-8-sig"

    )


    # ========================================================
    # 14. 总体统计
    # ========================================================

    total_complexes = len(
        qc_df
    )


    pass_total = int(
        (
            qc_df[
                "status"
            ]
            ==
            "PASS"
        ).sum()
    )


    fail_total = int(
        (
            qc_df[
                "status"
            ]
            ==
            "FAIL"
        ).sum()
    )


    exact_chain_match_total = int(
        qc_df[
            "exact_chain_match"
        ].sum()
    )


    overlap_total = int(
        (
            qc_df[
                "receptor_ligand_overlap"
            ]
            !=
            ""
        ).sum()
    )


    extra_chain_total = int(
        (
            qc_df[
                "unexpected_extra_chains"
            ]
            !=
            ""
        ).sum()
    )


    missing_expected_total = int(
        (
            qc_df[
                "missing_expected_chains"
            ]
            !=
            ""
        ).sum()
    )


    nonstandard_complex_total = int(
        (
            qc_df[
                "nonstandard_or_hetero_residue_count"
            ]
            >
            0
        ).sum()
    )


    nonstandard_residue_total = int(
        qc_df[
            "nonstandard_or_hetero_residue_count"
        ].sum()
    )


    # ========================================================
    # complex type统计
    # ========================================================

    type_counts = (

        qc_df[
            "complex_type"
        ]

        .value_counts()

        .to_dict()

    )


    single_single = int(
        type_counts.get(
            "single_receptor_single_ligand",
            0
        )
    )


    multi_rec_single_lig = int(
        type_counts.get(
            "multichain_receptor_single_ligand",
            0
        )
    )


    single_rec_multi_lig = int(
        type_counts.get(
            "single_receptor_multichain_ligand",
            0
        )
    )


    multi_multi = int(
        type_counts.get(
            "multichain_receptor_multichain_ligand",
            0
        )
    )


    multichain_total = (

        multi_rec_single_lig
        +
        single_rec_multi_lig
        +
        multi_multi

    )


    # ========================================================
    # split统计
    # ========================================================

    train_total = int(
        (
            qc_df[
                "split"
            ]
            ==
            "train"
        ).sum()
    )


    val_total = int(
        (
            qc_df[
                "split"
            ]
            ==
            "val"
        ).sum()
    )


    test_total = int(
        (
            qc_df[
                "split"
            ]
            ==
            "test"
        ).sum()
    )


    # ========================================================
    # Summary CSV
    # ========================================================

    summary_records = [

        {
            "metric":
                "Total complexes",
            "value":
                total_complexes
        },

        {
            "metric":
                "Train complexes",
            "value":
                train_total
        },

        {
            "metric":
                "Validation complexes",
            "value":
                val_total
        },

        {
            "metric":
                "Test complexes",
            "value":
                test_total
        },

        {
            "metric":
                "QC PASS",
            "value":
                pass_total
        },

        {
            "metric":
                "QC FAIL",
            "value":
                fail_total
        },

        {
            "metric":
                "Exact receptor+ligand chain match",
            "value":
                exact_chain_match_total
        },

        {
            "metric":
                "Missing expected-chain complexes",
            "value":
                missing_expected_total
        },

        {
            "metric":
                "Unexpected extra-chain complexes",
            "value":
                extra_chain_total
        },

        {
            "metric":
                "Receptor-ligand chain overlap complexes",
            "value":
                overlap_total
        },

        {
            "metric":
                "Complexes containing nonstandard/hetero residues",
            "value":
                nonstandard_complex_total
        },

        {
            "metric":
                "Total nonstandard/hetero residues",
            "value":
                nonstandard_residue_total
        },

        {
            "metric":
                "Single-chain receptor + single-chain ligand",
            "value":
                single_single
        },

        {
            "metric":
                "Multichain receptor + single-chain ligand",
            "value":
                multi_rec_single_lig
        },

        {
            "metric":
                "Single-chain receptor + multichain ligand",
            "value":
                single_rec_multi_lig
        },

        {
            "metric":
                "Multichain receptor + multichain ligand",
            "value":
                multi_multi
        },

        {
            "metric":
                "Complexes with at least one multichain partner",
            "value":
                multichain_total
        }

    ]


    summary_df = pd.DataFrame(
        summary_records
    )


    summary_df.to_csv(

        SUMMARY_CSV,

        index=False,

        encoding="utf-8-sig"

    )


    # ========================================================
    # 15. 最终打印
    # ========================================================

    print(
        "\n"
        +
        "=" * 78
    )


    print(
        "FINAL CHAIN PROCESSING QC"
    )


    print(
        "=" * 78
    )


    print(
        "\nDataset:"
    )


    print(
        "Train:",
        train_total
    )


    print(
        "Val:",
        val_total
    )


    print(
        "Test:",
        test_total
    )


    print(
        "Total:",
        total_complexes
    )


    print(
        "\nQC:"
    )


    print(
        "PASS:",
        pass_total
    )


    print(
        "FAIL:",
        fail_total
    )


    print(
        "Exact chain match:",
        exact_chain_match_total
    )


    print(
        "Missing expected-chain complexes:",
        missing_expected_total
    )


    print(
        "Unexpected extra-chain complexes:",
        extra_chain_total
    )


    print(
        "Receptor/ligand overlap complexes:",
        overlap_total
    )


    print(
        "Complexes with nonstandard/hetero residues:",
        nonstandard_complex_total
    )


    print(
        "Total nonstandard/hetero residues:",
        nonstandard_residue_total
    )


    print(
        "\nComplex composition:"
    )


    print(
        "Single receptor / Single ligand:",
        single_single
    )


    print(
        "Multi receptor / Single ligand:",
        multi_rec_single_lig
    )


    print(
        "Single receptor / Multi ligand:",
        single_rec_multi_lig
    )


    print(
        "Multi receptor / Multi ligand:",
        multi_multi
    )


    print(
        "At least one multichain partner:",
        multichain_total
    )


    print(
        "\nOutputs:"
    )


    print(
        "QC table:"
    )

    print(
        QC_CSV
    )


    print(
        "\nSummary:"
    )

    print(
        SUMMARY_CSV
    )


    print(
        "\nFailures:"
    )

    print(
        FAILURE_CSV
    )


    print(
        "\nMultichain complexes:"
    )

    print(
        MULTICHAIN_CSV
    )


    # ========================================================
    # 16. 理想状态提示
    # ========================================================

    if (
        fail_total == 0
        and
        exact_chain_match_total == total_complexes
        and
        missing_expected_total == 0
        and
        extra_chain_total == 0
        and
        overlap_total == 0
        and
        nonstandard_complex_total == 0
    ):


        print(
            "\n✅ DATASET-WIDE CHAIN QC PASSED"
        )


        print(
            "所有trimmed PDB均严格对应PDBbind指定的"
            " receptor + ligand chain groups。"
        )


        print(
            "没有缺失chain、额外chain、"
            "receptor/ligand重叠或非标准/hetero residue。"
        )


    else:


        print(
            "\n⚠️ CHAIN QC发现异常。"
        )


        print(
            "请先检查："
        )


        print(
            FAILURE_CSV
        )


    print(
        "=" * 78
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()