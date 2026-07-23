import os
import re
import pandas as pd
import shutil
from Bio.PDB import PDBParser

CLEAN_PDB_FOLDER = r"C:\Users\86198\Desktop\P-P\01_cleaned_protein_pdb"
INDEX_FILE = r"C:\Users\86198\Desktop\P-P\index\INDEX_general_PP.2020R1.txt"
FILTERED_PDB_FOLDER = r"C:\Users\86198\Desktop\P-P\02_filtered_final_pdb"
FINAL_CSV = r"C:\Users\86198\Desktop\P-P\filtered_pdb_metadata_final.csv"

# ====================== 筛选阈值 ======================
MAX_RESOLUTION = 3.0
MIN_CHAIN_LENGTH = 30

os.makedirs(FILTERED_PDB_FOLDER, exist_ok=True)


# ====================== 1. 解析INDEX文件：获取官方链标注和分辨率 ======================
def parse_index():
    pattern = re.compile(r"^([0-9a-z]{4})\s+(\d+\.?\d*).*?\((.*?)\|(.*?)\)", re.I)
    data = []
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            match = pattern.match(line)
            if match:
                data.append({
                    "pdb_id": match.group(1).lower(),
                    "resolution": float(match.group(2)),
                    "receptor_chain": match.group(3).strip(),
                    "ligand_chain": match.group(4).strip()
                })
    return pd.DataFrame(data)


# ====================== 2. 解析干净PDB：获取链数、链长 ======================
def get_pdb_info():
    parser = PDBParser(QUIET=True)
    rows = []
    for fname in os.listdir(CLEAN_PDB_FOLDER):
        if not fname.lower().endswith(".pdb"):
            continue
        pdb_id = fname[:4].lower()
        pdb_path = os.path.join(CLEAN_PDB_FOLDER, fname)
        try:
            structure = parser.get_structure(pdb_id, pdb_path)
            model = structure[0]
            chains = list(model)
            chain_count = len(chains)

            # 标准20种氨基酸
            STANDARD_AA = {
                'ALA', 'CYS', 'ASP', 'GLU', 'PHE',
                'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
                'MET', 'ASN', 'PRO', 'GLN', 'ARG',
                'SER', 'THR', 'VAL', 'TRP', 'TYR'
            }

            lengths = []
            for chain in chains:
                cnt = 0
                for res in chain:
                    if res.get_id()[0] == " " and res.get_resname() in STANDARD_AA:
                        cnt += 1
                if cnt > 0:
                    lengths.append(cnt)

            min_len = min(lengths) if lengths else 0
            max_len = max(lengths) if lengths else 0

            rows.append({
                "pdb_id": pdb_id,
                "pdb_file": fname,
                "pdb_path": pdb_path,
                "chain_count": chain_count,
                "min_chain_length": min_len,
                "max_chain_length": max_len
            })
        except:
            continue
    return pd.DataFrame(rows)


# ====================== 主流程 ======================
if __name__ == "__main__":
    # 1. 解析INDEX和PDB信息
    print("🔄 正在解析INDEX文件...")
    df_index = parse_index()

    print("🔄 正在解析PDB结构信息...")
    df_pdb = get_pdb_info()

    # 2. 合并数据
    print("🔄 正在合并数据并筛选...")
    df = pd.merge(df_pdb, df_index, on="pdb_id", how="inner")

    # 3. 核心筛选
    # - 分辨率 ≤ 3.0埃
    # - 必须有官方receptor_chain和ligand_chain标注
    # - 最小链长 ≥ 30残基
    df = df[
        (df["resolution"] <= MAX_RESOLUTION) &
        (df["receptor_chain"].notna()) &
        (df["ligand_chain"].notna()) &
        (df["receptor_chain"] != "") &
        (df["ligand_chain"] != "") &
        (df["min_chain_length"] >= MIN_CHAIN_LENGTH)
        ].copy()

    # 4. 复制筛选出的PDB到新文件夹
    print("🔄 正在复制筛选出的PDB...")
    copied_count = 0
    for idx, row in df.iterrows():
        src_path = row["pdb_path"]
        dst_path = os.path.join(FILTERED_PDB_FOLDER, row["pdb_file"])
        shutil.copy(src_path, dst_path)
        copied_count += 1

    # 5. 保存最终表格（按你要求的字段顺序）
    final_columns = [
        "pdb_id", "pdb_file", "pdb_path",
        "chain_count", "min_chain_length", "max_chain_length",
        "resolution", "receptor_chain", "ligand_chain"
    ]
    df = df[final_columns]
    df.to_csv(FINAL_CSV, index=False)

    # 6. 完成输出
    print("\n🎉 全部流程完成！")
    print(f"📊 最终有效蛋白-蛋白二聚体：{len(df)} 个")
    print(f"📂 筛选出的PDB已保存至：{FILTERED_PDB_FOLDER}")
    print(f"📂 最终表格保存至：{FINAL_CSV}")
    print(
        "✅ 表格包含字段：pdb_id, pdb_file, pdb_path, chain_count, min_chain_length, max_chain_length, resolution, receptor_chain, ligand_chain")