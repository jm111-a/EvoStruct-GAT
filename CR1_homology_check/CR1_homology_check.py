from Bio import SeqIO
from Bio import pairwise2
import pandas as pd
import warnings


# ================================
# 路径设置
# ================================

ROOT = r"C:\Users\Administrator\Desktop\P-P\CR1_homology_check"


QUERY_FILE = ROOT + r"\CR1_chain_C.fasta"


DATABASE_FILES = [
    ROOT + r"\receptor_sequences.fasta",
    ROOT + r"\ligand_sequences.fasta"
]


OUTPUT_FILE = ROOT + r"\CR1_homology_check_results.csv"


# ================================
# 读取CR1序列
# ================================

query_record = next(
    SeqIO.parse(
        QUERY_FILE,
        "fasta"
    )
)


query_id = query_record.id
query_seq = str(query_record.seq)

query_len = len(query_seq)


print("=" * 70)
print("CR1 query:")
print(query_id)
print("Length:", query_len)
print("=" * 70)



# ================================
# identity计算
# ================================

def calculate_identity(alignment):

    seqA = alignment.seqA
    seqB = alignment.seqB

    matches = 0
    aligned = 0


    for a, b in zip(seqA, seqB):

        if a == "-" or b == "-":
            continue

        aligned += 1

        if a == b:
            matches += 1


    if aligned == 0:
        return 0


    return matches / aligned * 100



# ================================
# 开始搜索
# ================================

results = []


for database in DATABASE_FILES:

    print("\nProcessing:")
    print(database)


    for record in SeqIO.parse(
        database,
        "fasta"
    ):

        target_id = record.id
        target_seq = str(record.seq)


        # local alignment
        alignment = pairwise2.align.localms(
            query_seq,
            target_seq,
            2,      # match
            -1,     # mismatch
            -5,     # gap open
            -0.5,   # gap extend
            one_alignment_only=True
        )[0]


        identity = calculate_identity(
            alignment
        )


        # query覆盖率
        aligned_query_length = sum(
            1
            for x in alignment.seqA
            if x != "-"
        )


        coverage = (
            aligned_query_length /
            query_len *
            100
        )


        # 高同源风险判断
        possible_leakage = (
            identity >= 50
            and
            coverage >= 50
        )


        results.append(
            {

                "database":
                    database.split("\\")[-1],

                "target":
                    target_id,

                "identity_%":
                    round(identity, 2),

                "coverage_%":
                    round(coverage, 2),

                "alignment_length":
                    aligned_query_length,

                "possible_leakage":
                    possible_leakage
            }
        )


# ================================
# 保存
# ================================

df = pd.DataFrame(results)


df = df.sort_values(
    by=[
        "identity_%",
        "coverage_%"
    ],
    ascending=False
)


df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)



print("\n" + "=" * 70)
print("Finished")
print("Saved:")
print(OUTPUT_FILE)
print("=" * 70)



print("\nTop 20 homologous proteins:")
print(
    df.head(20).to_string(
        index=False
    )
)



print("\nPotential leakage hits:")
print(
    df[
        df["possible_leakage"]
    ].to_string(
        index=False
    )
)