import os
import pandas as pd
from Bio.PDB import PDBParser


# =========================
# 路径
# =========================

ROOT = r"C:\Users\Administrator\Desktop\P-P"


PDB_DIR = os.path.join(
    ROOT,
    "03_chain_trimmed_pdb"
)


SPLIT_DIR = os.path.join(
    ROOT,
    "cluster_split_v3_A2",
    "04_split"
)


OUT_DIR = os.path.join(
    ROOT,
    "cluster_split_v3_A2",
    "05_labels"
)


DIST_THRESHOLD = 5.0


for s in ["train","val","test"]:
    os.makedirs(
        os.path.join(OUT_DIR,s),
        exist_ok=True
    )


parser = PDBParser(QUIET=True)



# =========================
# heavy atoms
# =========================

def get_heavy_atoms(res):

    return [
        atom for atom in res
        if atom.element != "H"
    ]



def residue_distance(res1,res2):

    atoms1=get_heavy_atoms(res1)
    atoms2=get_heavy_atoms(res2)

    min_d=999


    for a1 in atoms1:
        for a2 in atoms2:

            d=a1-a2

            if d < min_d:
                min_d=d

            if min_d < DIST_THRESHOLD:
                return min_d


    return min_d



# =========================
# 单个PDB
# =========================


def generate_label(
        pdb_file,
        rec_chain,
        lig_chain):


    pdb_path=os.path.join(
        PDB_DIR,
        pdb_file
    )


    structure=parser.get_structure(
        "x",
        pdb_path
    )

    model=structure[0]


    rec_res=[]
    lig_res=[]


    # receptor

    for c in str(rec_chain):

        if c not in model:
            continue


        for r in model[c]:

            if r.get_id()[0]==" ":
                rec_res.append(
                    (c,r)
                )


    # ligand

    for c in str(lig_chain):

        if c not in model:
            continue


        for r in model[c]:

            if r.get_id()[0]==" ":
                lig_res.append(
                    (c,r)
                )



    labels=[]



    # receptor label

    for c,r in rec_res:

        y=0

        for _,lr in lig_res:

            if residue_distance(r,lr)<DIST_THRESHOLD:

                y=1
                break


        labels.append({

            "chain_id":c,
            "residue_index":r.get_id()[1],
            "residue_name":r.get_resname(),
            "interface_label":y,
            "chain_type":"receptor"

        })



    # ligand label

    for c,r in lig_res:

        y=0

        for _,rr in rec_res:

            if residue_distance(r,rr)<DIST_THRESHOLD:

                y=1
                break


        labels.append({

            "chain_id":c,
            "residue_index":r.get_id()[1],
            "residue_name":r.get_resname(),
            "interface_label":y,
            "chain_type":"ligand"

        })


    return labels




# =========================
# split处理
# =========================


for split in ["train","val","test"]:


    print("\n==========",split,"==========")


    meta=pd.read_csv(
        os.path.join(
            SPLIT_DIR,
            f"{split}_metadata.csv"
        )
    )


    success=0
    fail=0


    for _,row in meta.iterrows():


        pdb=row["pdb_file"]


        try:

            labels=generate_label(

                pdb,

                row["receptor_chain"],

                row["ligand_chain"]

            )


            out=os.path.join(
                OUT_DIR,
                split,
                pdb.replace(
                    ".pdb",
                    "_label.csv"
                )
            )


            pd.DataFrame(labels).to_csv(
                out,
                index=False
            )


            success+=1


        except Exception as e:

            print(
                "FAILED",
                pdb,
                e
            )

            fail+=1



    print(
        "成功:",
        success,
        "失败:",
        fail
    )


print("\n全部完成")