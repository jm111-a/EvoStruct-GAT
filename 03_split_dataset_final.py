import os
import pandas as pd
import random
import shutil

TRIMMED_PDB_FOLDER = r"C:\Users\86198\Desktop\P-P\03_chain_trimmed_pdb"
METADATA_CSV = r"C:\Users\86198\Desktop\P-P\filtered_pdb_metadata_final.csv"
OUTPUT_ROOT = r"C:\Users\86198\Desktop\P-P\dataset_split"

N_TRAIN = 602
N_VAL = 174
N_TEST = 175
SEED = 42

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_ROOT, split), exist_ok=True)

df = pd.read_csv(METADATA_CSV)
available_pdbs = set(os.listdir(TRIMMED_PDB_FOLDER))
df = df[df["pdb_file"].isin(available_pdbs)].copy()
pdb_list = df["pdb_file"].tolist()
print(f"📊 总有效蛋白-蛋白二聚体：{len(pdb_list)} 个")

expected_total = N_TRAIN + N_VAL + N_TEST
if len(pdb_list) != expected_total:
    print(f"⚠️ 警告：当前读取到的有效 PDB 总数（{len(pdb_list)}）与设定的切分总数（{expected_total}）不一致！")
    print("代码将尽可能按照设定数量切分，可能会有数据剩余或报错，请仔细核对前置步骤。")

random.seed(SEED)
random.shuffle(pdb_list)

train_files = pdb_list[:N_TRAIN]
val_files = pdb_list[N_TRAIN : N_TRAIN + N_VAL]
test_files = pdb_list[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]

def copy_files(file_list, split_name):
    dst_folder = os.path.join(OUTPUT_ROOT, split_name)
    copied = 0
    for fname in file_list:
        src = os.path.join(TRIMMED_PDB_FOLDER, fname)
        dst = os.path.join(dst_folder, fname)
        shutil.copy(src, dst)
        copied += 1
    print(f"✅ {split_name}：复制完成 {copied} 个PDB")

copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

df["split"] = df["pdb_file"].apply(
    lambda x: "train" if x in train_files else "val" if x in val_files else "test" if x in test_files else "unused"
)
split_record_path = os.path.join(OUTPUT_ROOT, "dataset_split_record.csv")
df.to_csv(split_record_path, index=False)

print("\n🎉 数据集划分完成！")
print(f"📂 划分结果保存至：{OUTPUT_ROOT}")
print(f"   ├─ train：{len(train_files)} 个")
print(f"   ├─ val：{len(val_files)} 个")
print(f"   └─ test：{len(test_files)} 个")
print(f"📋 划分记录已保存至：{split_record_path}")