from pathlib import Path
import pandas as pd
from Bio.PDB import PDBParser


# ======================================================
# 路径
# ======================================================

BASE_DIR = Path(
    r"C:\Users\86198\Desktop\P-P"
)


INPUT_META = (
    BASE_DIR /
    "filtered_pdb_metadata_final.csv"
)


PDB_DIR = (
    BASE_DIR /
    "03_chain_trimmed_pdb"
)


FINAL_META = (
    BASE_DIR /
    "final_metadata.csv"
)


FAILED_META = (
    BASE_DIR /
    "failed_complex.csv"
)



# ======================================================
# 初始化
# ======================================================

parser = PDBParser(
    QUIET=True
)


valid_records = []

failed_records = []



# ======================================================
# 检查链
# ======================================================

def check_chain_exists(model, chain_ids):

    chain_ids = str(chain_ids)

    for cid in chain_ids:

        if cid not in model:

            return False

    return True



# ======================================================
# 主程序
# ======================================================


print("="*70)
print("Dataset QC")
print("="*70)



metadata = pd.read_csv(
    INPUT_META
)


print(
    "Input complexes:",
    len(metadata)
)



for _, row in metadata.iterrows():


    pdb_file = row["pdb_file"]


    pdb_path = (
        PDB_DIR /
        pdb_file
    )


    # -------------------------
    # pdb存在
    # -------------------------

    if not pdb_path.exists():

        failed_records.append({

            "pdb_file": pdb_file,

            "reason":
            "PDB file missing",

            "receptor_chain":
            row["receptor_chain"],

            "ligand_chain":
            row["ligand_chain"],

            "available_chains":
            ""

        })

        continue



    try:


        structure = parser.get_structure(
            pdb_file,
            pdb_path
        )


        model = structure[0]


        available = ",".join(
            [
                c.id
                for c in model
            ]
        )


        # -------------------------
        # receptor
        # -------------------------

        rec_ok = check_chain_exists(
            model,
            row["receptor_chain"]
        )


        # -------------------------
        # ligand
        # -------------------------

        lig_ok = check_chain_exists(
            model,
            row["ligand_chain"]
        )



        if not rec_ok:


            failed_records.append({

                "pdb_file": pdb_file,

                "reason":
                "missing receptor chain",

                "receptor_chain":
                row["receptor_chain"],

                "ligand_chain":
                row["ligand_chain"],

                "available_chains":
                available

            })

            continue



        if not lig_ok:


            failed_records.append({

                "pdb_file": pdb_file,

                "reason":
                "missing ligand chain",

                "receptor_chain":
                row["receptor_chain"],

                "ligand_chain":
                row["ligand_chain"],

                "available_chains":
                available

            })

            continue



        # -------------------------
        # 保留
        # -------------------------

        valid_records.append(
            row.to_dict()
        )



    except Exception as e:


        failed_records.append({

            "pdb_file": pdb_file,

            "reason":
            str(e),

            "receptor_chain":
            row["receptor_chain"],

            "ligand_chain":
            row["ligand_chain"],

            "available_chains":
            ""

        })



# ======================================================
# 保存
# ======================================================


final_df = pd.DataFrame(
    valid_records
)


failed_df = pd.DataFrame(
    failed_records
)



final_df.to_csv(
    FINAL_META,
    index=False
)



failed_df.to_csv(
    FAILED_META,
    index=False
)



print("\n"+"="*70)

print(
    "QC finished"
)


print(
    "Original:",
    len(metadata)
)


print(
    "Valid:",
    len(final_df)
)


print(
    "Failed:",
    len(failed_df)
)


print(
    "\nFinal metadata:",
    FINAL_META
)


print(
    "Failed list:",
    FAILED_META
)


if len(failed_df)>0:

    print("\nFailed complexes:")
    print(failed_df)


print("="*70)