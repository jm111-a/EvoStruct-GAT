from pathlib import Path
import pandas as pd
import re


# =====================================================
# 路径
# =====================================================

BASE_DIR = Path(
    r"C:\Users\86198\Desktop\P-P"
)


CDHIT_DIR = (
    BASE_DIR /
    "cluster_split_v3_A2" /
    "02_cdhit"
)


META_FILE = (
    BASE_DIR /
    "cluster_split_v3_A2" /
    "01_complex_cluster_input" /
    "complex_sequence_metadata.csv"
)


OUT_DIR = (
    BASE_DIR /
    "cluster_split_v3_A2" /
    "03_complex_cluster_mapping"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OUT_MAPPING = (
    OUT_DIR /
    "complex_cluster_mapping.csv"
)


OUT_SUMMARY = (
    OUT_DIR /
    "cluster_summary.csv"
)



# =====================================================
# 解析CD-HIT clstr
# =====================================================


def parse_cdhit_cluster(clstr_file, prefix):

    """
    输出:
    complex_id -> cluster_id
    """

    mapping = {}

    cluster_id = None


    with open(
        clstr_file,
        "r"
    ) as f:


        for line in f:


            line=line.strip()


            if line.startswith(">Cluster"):


                num = (
                    line
                    .replace(">Cluster ","")
                )

                cluster_id = (
                    prefix
                    +
                    f"{int(num):04d}"
                )


            elif line:


                # 找序列名字
                m = re.search(
                    r">(.+?)\.\.\.",
                    line
                )


                if m:

                    seq_id = m.group(1)

                    mapping[
                        seq_id
                    ] = cluster_id


    return mapping



# =====================================================
# 主程序
# =====================================================


print("="*70)

print(
    "解析CD-HIT cluster"
)

print("="*70)



receptor_map = parse_cdhit_cluster(
    CDHIT_DIR /
    "receptor_40.fasta.clstr",
    "R"
)


ligand_map = parse_cdhit_cluster(
    CDHIT_DIR /
    "ligand_40.fasta.clstr",
    "L"
)



print(
    "receptor mapping:",
    len(receptor_map)
)


print(
    "ligand mapping:",
    len(ligand_map)
)



metadata = pd.read_csv(
    META_FILE
)



records=[]



for _,row in metadata.iterrows():


    cid=row["complex_id"]


    r_cluster = receptor_map.get(
        cid,
        None
    )


    l_cluster = ligand_map.get(
        cid,
        None
    )


    if r_cluster is None or l_cluster is None:

        print(
            "missing cluster:",
            cid
        )

        continue



    records.append(
        {

        "complex_id":cid,

        "pdb_file":
            row["pdb_file"],

        "receptor_cluster":
            r_cluster,

        "ligand_cluster":
            l_cluster,

        "complex_cluster":
            r_cluster
            +
            "_"
            +
            l_cluster

        }
    )



df=pd.DataFrame(records)



df.to_csv(
    OUT_MAPPING,
    index=False
)



summary = (
    df
    .groupby(
        "complex_cluster"
    )
    .size()
    .reset_index(
        name="complex_number"
    )
    .sort_values(
        "complex_number",
        ascending=False
    )
)


summary.to_csv(
    OUT_SUMMARY,
    index=False
)



print("\n完成")

print(
    "complex数量:",
    len(df)
)


print(
    "complex cluster数量:",
    df["complex_cluster"].nunique()
)


print(
    "最大cluster:",
    summary.iloc[0].to_dict()
)


print(
    OUT_MAPPING
)
