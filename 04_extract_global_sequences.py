import os
from Bio.PDB import PDBParser

PDB_INPUT_FOLDER = r"C:\Users\86198\Desktop\P-P\03_chain_trimmed_pdb"
OUTPUT_FASTA = r"C:\Users\86198\Desktop\P-P\cdhit_global_input.fasta"
parser = PDBParser(QUIET=True)
# 标准氨基酸 3字母转1字母 字典
aa_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


# ====================== 主流程 ======================
def main():
    fasta_content = ""
    success_count = 0
    fail_count = 0

    print(f"🔄 正在从 {PDB_INPUT_FOLDER} 提取全局PDB序列...")

    for fname in os.listdir(PDB_INPUT_FOLDER):
        if not fname.lower().endswith(".pdb"):
            continue

        pdb_id = fname[:4].lower()
        pdb_path = os.path.join(PDB_INPUT_FOLDER, fname)

        try:
            structure = parser.get_structure(pdb_id, pdb_path)
            model = structure[0]
            full_seq = ""

            for chain in model:
                for res in chain:
                    if res.get_id()[0] == " " and res.get_resname() in aa_map:
                        full_seq += aa_map[res.get_resname()]
            if len(full_seq) >= 30:
                fasta_content += f">{pdb_id}\n{full_seq}\n"
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            print(f"⚠️ 提取失败 {fname}: {e}")
            fail_count += 1

    with open(OUTPUT_FASTA, "w", encoding="utf-8") as f:
        f.write(fasta_content)

    print("\n🎉 全局序列提取完成！")
    print(f"✅ 成功提取序列：{success_count} 条")
    print(f"❌ 失败或过短：{fail_count} 条")
    print(f"📂 输出的全局 FASTA 文件：{OUTPUT_FASTA}")


if __name__ == "__main__":
    main()