import os
import re
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select

RAW_PDB_FOLDER = r"C:\Users\86198\Desktop\P-P\00_raw_pdb"
INDEX_FILE = r"C:\Users\86198\Desktop\P-P\index\INDEX_general_PP.2020R1.txt"
CLEAN_PDB_FOLDER = r"C:\Users\86198\Desktop\P-P\01_cleaned_protein_pdb"
FINAL_CSV = r"C:\Users\86198\Desktop\P-P\filtered_pdb_metadata_final.csv"

# ====================== 筛选阈值 ======================
MAX_RESOLUTION = 3
MIN_CHAIN_LENGTH = 30

os.makedirs(CLEAN_PDB_FOLDER, exist_ok=True)


# ====================== 1. 去水去杂：仅保留标准氨基酸 ======================
class ProteinOnlySelect(Select):
    STANDARD_AA = {
        'ALA', 'CYS', 'ASP', 'GLU', 'PHE',
        'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
        'MET', 'ASN', 'PRO', 'GLN', 'ARG',
        'SER', 'THR', 'VAL', 'TRP', 'TYR'
    }

    def accept_residue(self, residue):
        if residue.get_id()[0] == " " and residue.get_resname() in self.STANDARD_AA:
            return True
        return False

    def accept_chain(self, chain):
        for residue in chain:
            if self.accept_residue(residue):
                return True
        return False


def clean_pdb_files():
    parser = PDBParser(QUIET=True)
    io = PDBIO()
    success = 0
    for fname in os.listdir(RAW_PDB_FOLDER):
        if not fname.lower().endswith(".pdb"):
            continue
        pdb_id = fname[:4].lower()
        raw_path = os.path.join(RAW_PDB_FOLDER, fname)
        clean_path = os.path.join(CLEAN_PDB_FOLDER, fname)
        try:
            structure = parser.get_structure(pdb_id, raw_path)
            io.set_structure(structure)
            io.save(clean_path, ProteinOnlySelect())
            success += 1
        except:
            continue
    print(f"✅ 去水完成：成功清理 {success} 个PDB")


# ====================== 2. 解析INDEX文件：获取官方链标注 ======================
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


# ====================== 3. 解析干净PDB：获取链数、链长 ======================
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

            aa_map = ProteinOnlySelect.STANDARD_AA
            lengths = []
            for chain in chains:
                cnt = 0
                for res in chain:
                    if res.get_id()[0] == " " and res.get_resname() in aa_map:
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


# ====================== 主流程：一步生成最终表格 ======================
if __name__ == "__main__":
    # 1. 去水
    print("🔄 正在去水去杂...")
    clean_pdb_files()

    # 2. 解析INDEX
    print("🔄 正在解析INDEX文件...")
    df_index = parse_index()

    # 3. 解析PDB信息
    print("🔄 正在解析PDB结构信息...")
    df_pdb = get_pdb_info()

    # 4. 合并 + 筛选
    print("🔄 正在合并数据并筛选...")
    df = pd.merge(df_pdb, df_index, on="pdb_id", how="inner")

    # 核心筛选：必须有官方链标注 + 分辨率≤2.5 + 最小链长≥30
    df = df[
        (df["receptor_chain"].notna()) &
        (df["ligand_chain"].notna()) &
        (df["receptor_chain"] != "") &
        (df["ligand_chain"] != "") &
        (df["resolution"] <= MAX_RESOLUTION) &
        (df["min_chain_length"] >= MIN_CHAIN_LENGTH)
        ].copy()

    # 5. 保存最终表格（按你要求的字段顺序）
    final_columns = [
        "pdb_id", "pdb_file", "pdb_path",
        "chain_count", "min_chain_length", "max_chain_length",
        "resolution", "receptor_chain", "ligand_chain"
    ]
    df = df[final_columns]
    df.to_csv(FINAL_CSV, index=False)

    print("\n🎉 全部流程完成！")
    print(f"📊 最终有效蛋白-蛋白二聚体：{len(df)} 个")
    print(f"📂 最终表格保存至：{FINAL_CSV}")
    print(
        "✅ 包含字段：pdb_id, pdb_file, pdb_path, chain_count, min_chain_length, max_chain_length, resolution, receptor_chain, ligand_chain")