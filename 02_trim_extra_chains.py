import os
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select

PDB_INPUT_FOLDER = r"C:\Users\86198\Desktop\P-P\02_filtered_final_pdb"
METADATA_CSV = r"C:\Users\86198\Desktop\P-P\filtered_pdb_metadata_final.csv"
PDB_OUTPUT_FOLDER = r"C:\Users\86198\Desktop\P-P\03_chain_trimmed_pdb"

os.makedirs(PDB_OUTPUT_FOLDER, exist_ok=True)

# ====================== 1. 读取元数据，获取每个PDB要保留的链 ======================
df = pd.read_csv(METADATA_CSV)
# 构建字典：pdb文件名 → 要保留的链列表（受体链+配体链）
pdb_keep_chains = {}
for idx, row in df.iterrows():
    pdb_file = row["pdb_file"]
    # 拆分受体链和配体链，支持多链（如HL、A,B）
    rec_chains = list(str(row["receptor_chain"]).replace(",", "").replace(" ", ""))
    lig_chains = list(str(row["ligand_chain"]).replace(",", "").replace(" ", ""))
    # 合并去重，得到所有要保留的链
    keep_chains = list(set(rec_chains + lig_chains))
    pdb_keep_chains[pdb_file] = keep_chains

# ====================== 2. 定义链筛选规则：只保留指定的链 ======================
class KeepTargetChains(Select):
    def __init__(self, target_chains):
        self.target_chains = target_chains

    def accept_chain(self, chain):
        # 只保留在目标列表里的链
        return chain.get_id() in self.target_chains

# ====================== 3. 批量处理PDB ======================
parser = PDBParser(QUIET=True)
io = PDBIO()
success_count = 0
fail_count = 0

for pdb_file in os.listdir(PDB_INPUT_FOLDER):
    if not pdb_file.lower().endswith(".pdb"):
        continue
    if pdb_file not in pdb_keep_chains:
        print(f"⚠️  跳过：{pdb_file} 不在元数据表格中")
        fail_count += 1
        continue

    # 获取当前PDB要保留的链
    target_chains = pdb_keep_chains[pdb_file]
    input_path = os.path.join(PDB_INPUT_FOLDER, pdb_file)
    output_path = os.path.join(PDB_OUTPUT_FOLDER, pdb_file)

    try:
        # 解析PDB
        structure = parser.get_structure(pdb_file[:4], input_path)
        # 筛选链并保存
        io.set_structure(structure)
        io.save(output_path, KeepTargetChains(target_chains))
        print(f"✅ 处理完成：{pdb_file} | 保留链：{target_chains}")
        success_count += 1
    except Exception as e:
        print(f"❌ 处理失败：{pdb_file} | 错误：{str(e)}")
        fail_count += 1

# ====================== 最终统计 ======================
print("\n🎉 全部处理完成！")
print(f"✅ 成功处理：{success_count} 个PDB")
print(f"❌ 失败/跳过：{fail_count} 个PDB")
print(f"📂 处理后的干净PDB已保存至：{PDB_OUTPUT_FOLDER}")