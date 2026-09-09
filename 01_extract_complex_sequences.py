from pathlib import Path
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1


# ======================================================
# 路径配置
# ======================================================

BASE_DIR = Path(
    r"C:\Users\86198\Desktop\P-P"
)


METADATA_FILE = (
    BASE_DIR /
    "final_metadata.csv"
)


PDB_DIR = (
    BASE_DIR /
    "03_chain_trimmed_pdb"
)


OUT_DIR = (
    BASE_DIR /
    "cluster_split_v3_A2" /
    "01_complex_cluster_input"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



RECEPTOR_FASTA = OUT_DIR / "receptor_sequences.fasta"

LIGAND_FASTA = OUT_DIR / "ligand_sequences.fasta"

META_OUTPUT = OUT_DIR / "complex_sequence_metadata.csv"

FAILED_OUTPUT = OUT_DIR / "failed_complex.csv"



# ======================================================
# 提取链序列
# ======================================================

def extract_chain_sequence(structure, chain_ids):

    sequence = ""

    chain_ids = str(chain_ids)


    available_chains = [
        c.id for c in structure[0]
    ]


    for cid in chain_ids:


        found = False


        for chain in structure[0]:


            if chain.id == cid:

                found = True


                for residue in chain:


                    if residue.id[0] == " ":

                        try:

                            aa = seq1(
                                residue.resname
                            )

                            if aa != "X":

                                sequence += aa

                        except:

                            pass


                break


        if not found:

            return None


    return sequence



# ======================================================
# 主程序
# ======================================================


print("="*70)
print("Extract receptor / ligand complex sequences")
print("="*70)



metadata = pd.read_csv(
    METADATA_FILE
)


print(
    "metadata complex数量:",
    len(metadata)
)



parser = PDBParser(
    QUIET=True
)



receptor_records = []

ligand_records = []

metadata_records = []

failed_records = []

success = 0



for idx,row in metadata.iterrows():


    pdb_file = row["pdb_file"]


    pdb_path = (
        PDB_DIR /
        pdb_file
    )



    if not pdb_path.exists():

        failed_records.append({

            "pdb_file": pdb_file,

            "receptor_chain":
                row["receptor_chain"],

            "ligand_chain":
                row["ligand_chain"],

            "reason":
                "PDB file missing",

            "available_chains":
                ""

        })

        continue



    try:


        structure = parser.get_structure(
            pdb_file,
            pdb_path
        )


        available_chains = ",".join(
            [
                c.id
                for c in structure[0]
            ]
        )



        receptor_chain = row[
            "receptor_chain"
        ]

        ligand_chain = row[
            "ligand_chain"
        ]



        receptor_seq = extract_chain_sequence(
            structure,
            receptor_chain
        )


        ligand_seq = extract_chain_sequence(
            structure,
            ligand_chain
        )



        if receptor_seq is None:


            failed_records.append({

                "pdb_file": pdb_file,

                "receptor_chain":
                    receptor_chain,

                "ligand_chain":
                    ligand_chain,

                "reason":
                    "missing receptor chain",

                "available_chains":
                    available_chains

            })

            continue



        if ligand_seq is None:


            failed_records.append({

                "pdb_file": pdb_file,

                "receptor_chain":
                    receptor_chain,

                "ligand_chain":
                    ligand_chain,

                "reason":
                    "missing ligand chain",

                "available_chains":
                    available_chains

            })

            continue



        if len(receptor_seq)==0:


            failed_records.append({

                "pdb_file": pdb_file,

                "receptor_chain":
                    receptor_chain,

                "ligand_chain":
                    ligand_chain,

                "reason":
                    "empty receptor sequence",

                "available_chains":
                    available_chains

            })

            continue



        if len(ligand_seq)==0:


            failed_records.append({

                "pdb_file": pdb_file,

                "receptor_chain":
                    receptor_chain,

                "ligand_chain":
                    ligand_chain,

                "reason":
                    "empty ligand sequence",

                "available_chains":
                    available_chains

            })

            continue




        complex_id = pdb_file.replace(
            ".pdb",
            ""
        )



        receptor_records.append(
            (
                complex_id,
                receptor_seq
            )
        )


        ligand_records.append(
            (
                complex_id,
                ligand_seq
            )
        )



        metadata_records.append({

            "complex_id":
                complex_id,

            "pdb_file":
                pdb_file,

            "receptor_chain":
                receptor_chain,

            "ligand_chain":
                ligand_chain,

            "receptor_length":
                len(receptor_seq),

            "ligand_length":
                len(ligand_seq)

        })


        success += 1



    except Exception as e:


        failed_records.append({

            "pdb_file": pdb_file,

            "receptor_chain":
                row["receptor_chain"],

            "ligand_chain":
                row["ligand_chain"],

            "reason":
                str(e),

            "available_chains":
                ""

        })



# ======================================================
# 输出FASTA
# ======================================================


with open(
    RECEPTOR_FASTA,
    "w"
) as f:


    for name,seq in receptor_records:

        f.write(
            f">{name}\n"
        )

        f.write(
            seq+"\n"
        )




with open(
    LIGAND_FASTA,
    "w"
) as f:


    for name,seq in ligand_records:

        f.write(
            f">{name}\n"
        )

        f.write(
            seq+"\n"
        )




pd.DataFrame(
    metadata_records
).to_csv(
    META_OUTPUT,
    index=False
)



pd.DataFrame(
    failed_records
).to_csv(
    FAILED_OUTPUT,
    index=False
)



# ======================================================
# summary
# ======================================================


print("\n"+"="*70)

print("完成")

print(
    "成功complex:",
    success
)

print(
    "失败complex:",
    len(failed_records)
)


print(
    "receptor fasta:",
    RECEPTOR_FASTA
)


print(
    "ligand fasta:",
    LIGAND_FASTA
)


print(
    "metadata:",
    META_OUTPUT
)


print(
    "failed:",
    FAILED_OUTPUT
)


if len(failed_records)>0:

    print("\n失败列表:")
    print(
        pd.DataFrame(
            failed_records
        )
    )


print("="*70)