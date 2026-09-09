import os
import random
import pandas as pd


# ============================================================
# 路径
# ============================================================

ROOT = r"C:\Users\86198\Desktop\P-P"


# complex cluster mapping
CLUSTER_FILE = os.path.join(
    ROOT,
    "cluster_split_v3_A2",
    "03_complex_cluster_mapping",
    "complex_cluster_mapping.csv"
)


# metadata
METADATA_FILE = os.path.join(
    ROOT,
    "final_metadata.csv"
)


# 输出
OUT_DIR = os.path.join(
    ROOT,
    "cluster_split_v3_A2",
    "04_split"
)


os.makedirs(
    OUT_DIR,
    exist_ok=True
)



SEED = 42



random.seed(
    SEED
)



# ============================================================
# 读取
# ============================================================


print("="*70)

print(
    "Strict receptor/ligand cluster split"
)

print("="*70)



cluster_df = pd.read_csv(
    CLUSTER_FILE
)


meta_df = pd.read_csv(
    METADATA_FILE
)



print(
    "cluster:",
    cluster_df.shape
)


print(
    "metadata:",
    meta_df.shape
)



# merge

df = pd.merge(

    meta_df,

    cluster_df,

    left_on="pdb_file",

    right_on="pdb_file",

    how="inner"

)



print(
    "merged:",
    df.shape
)



# ============================================================
# 构建 cluster block
# ============================================================


# 一个block代表：
# 任何共享 receptor_cluster 或 ligand_cluster 的complex
# 必须放在同一个split


parent = {}


def find(x):

    if parent[x] != x:

        parent[x] = find(
            parent[x]
        )

    return parent[x]



def union(a,b):

    ra=find(a)
    rb=find(b)

    if ra != rb:

        parent[rb]=ra



complex_ids = list(
    df.index
)



for i in complex_ids:

    parent[i]=i



# receptor cluster连接

rec_map={}


for idx,row in df.iterrows():

    rc=row[
        "receptor_cluster"
    ]

    if rc in rec_map:

        union(
            idx,
            rec_map[rc]
        )

    else:

        rec_map[rc]=idx



# ligand cluster连接

lig_map={}


for idx,row in df.iterrows():

    lc=row[
        "ligand_cluster"
    ]

    if lc in lig_map:

        union(
            idx,
            lig_map[lc]
        )

    else:

        lig_map[lc]=idx




# ============================================================
# 得到cluster blocks
# ============================================================


blocks={}


for idx in df.index:

    root=find(idx)

    blocks.setdefault(
        root,
        []
    ).append(idx)



cluster_blocks=list(
    blocks.values()
)



print(
    "Strict blocks:",
    len(cluster_blocks)
)



block_sizes=[
    len(x)
    for x in cluster_blocks
]


print(
    "Largest block:",
    max(block_sizes)
)



# ============================================================
# shuffle blocks
# ============================================================


random.shuffle(
    cluster_blocks
)



# ============================================================
# 按complex数量分配
# ============================================================


total=len(df)


target={

    "train":
        int(total*0.8),

    "val":
        int(total*0.1),

    "test":
        total -
        int(total*0.8)
        -
        int(total*0.1)

}



print(
    "Target:",
    target
)



splits={

    "train":[],

    "val":[],

    "test":[]

}



counts={

    "train":0,

    "val":0,

    "test":0

}



for block in cluster_blocks:


    # 当前最缺的split

    remain={

        k:
        target[k]-counts[k]

        for k in target

    }


    split=max(
        remain,
        key=remain.get
    )


    splits[split].extend(
        block
    )


    counts[split]+=len(block)




print(
    "\nFinal counts:"
)

print(
    counts
)




# ============================================================
# 输出metadata
# ============================================================


for split,idxs in splits.items():


    out=df.loc[
        idxs
    ].copy()


    out=out.reset_index(
        drop=True
    )


    outfile=os.path.join(

        OUT_DIR,

        f"{split}_metadata.csv"

    )


    out.to_csv(

        outfile,

        index=False

    )


    print(

        split,

        len(out),

        outfile

    )




# ============================================================
# 验证泄漏
# ============================================================


train=pd.read_csv(
    os.path.join(
        OUT_DIR,
        "train_metadata.csv"
    )
)


val=pd.read_csv(
    os.path.join(
        OUT_DIR,
        "val_metadata.csv"
    )
)


test=pd.read_csv(
    os.path.join(
        OUT_DIR,
        "test_metadata.csv"
    )
)



def overlap(
    a,b,col
):

    return len(
        set(a[col])
        &
        set(b[col])
    )



print("\nLeakage check")



for col in [

    "complex_cluster",

    "receptor_cluster",

    "ligand_cluster"

]:


    print(

        col,

        "train-test:",
        overlap(
            train,
            test,
            col
        ),

        "train-val:",
        overlap(
            train,
            val,
            col
        ),

        "val-test:",
        overlap(
            val,
            test,
            col
        )

    )



print("\nFinished")

print(
    OUT_DIR
)
