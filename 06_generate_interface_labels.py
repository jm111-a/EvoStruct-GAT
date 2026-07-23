import os
import pandas as pd
from Bio.PDB import PDBParser

METADATA_CSV = r"C:\Users\Administrator\Desktop\P-P\filtered_pdb_metadata_final.csv"
DATASET_ROOT = r"C:\Users\Administrator\Desktop\P-P\dataset_split"
OUTPUT_LABEL_ROOT = r"C:\Users\Administrator\Desktop\P-P\labels"

DIST_THRESHOLD = 5.0

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_LABEL_ROOT, split), exist_ok=True)

df_meta = pd.read_csv(METADATA_CSV)
pdb2rec = dict(zip(df_meta["pdb_file"], df_meta["receptor_chain"]))
pdb2lig = dict(zip(df_meta["pdb_file"], df_meta["ligand_chain"]))

def get_heavy_atoms(residue):
    return [atom for atom in residue if atom.element != "H"]

def min_residue_distance(res1, res2):
    min_dist = 999.0
    atoms1 = get_heavy_atoms(res1)
    atoms2 = get_heavy_atoms(res2)
    if not atoms1 or not atoms2:
        return min_dist
    for a1 in atoms1:
        for a2 in atoms2:
            dist = a1 - a2
            if dist < min_dist:
                min_dist = dist
            if min_dist < DIST_THRESHOLD:
                return min_dist
    return min_dist

def generate_single_pdb_label(pdb_path, rec_chain_str, lig_chain_str):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("tmp", pdb_path)
        model = structure[0]
    except:
        return None

    rec_chains = list(str(rec_chain_str).replace(",", "").replace(" ", ""))
    lig_chains = list(str(lig_chain_str).replace(",", "").replace(" ", ""))

    rec_residues = []
    for ch_id in rec_chains:
        if ch_id not in model:
            continue
        for res in model[ch_id]:
            if res.get_id()[0] == " ":
                rec_residues.append((ch_id, res))

    lig_residues = []
    for ch_id in lig_chains:
        if ch_id not in model:
            continue
        for res in model[ch_id]:
            if res.get_id()[0] == " ":
                lig_residues.append((ch_id, res))

    if not rec_residues or not lig_residues:
        return None

    label_data = []

    for r_ch_id, r_res in rec_residues:
        is_interface = 0
        for l_ch_id, l_res in lig_residues:
            if min_residue_distance(r_res, l_res) < DIST_THRESHOLD:
                is_interface = 1
                break
        label_data.append({
            "chain_id": r_ch_id,
            "residue_index": r_res.get_id()[1],
            "residue_name": r_res.get_resname(),
            "interface_label": is_interface,
            "chain_type": "receptor"
        })

    for l_ch_id, l_res in lig_residues:
        is_interface = 0
        for r_ch_id, r_res in rec_residues:
            if min_residue_distance(l_res, r_res) < DIST_THRESHOLD:
                is_interface = 1
                break
        label_data.append({
            "chain_id": l_ch_id,
            "residue_index": l_res.get_id()[1],
            "residue_name": l_res.get_resname(),
            "interface_label": is_interface,
            "chain_type": "ligand"
        })

    return label_data

def process_split(split_name):
    pdb_folder = os.path.join(DATASET_ROOT, split_name)
    output_folder = os.path.join(OUTPUT_LABEL_ROOT, split_name)
    success_count = 0
    fail_count = 0

    for pdb_file in os.listdir(pdb_folder):
        if not pdb_file.lower().endswith(".pdb"):
            continue
        if pdb_file not in pdb2rec:
            print(f"⚠️  跳过：{pdb_file} 不在元数据中")
            fail_count += 1
            continue

        rec_chain = pdb2rec[pdb_file]
        lig_chain = pdb2lig[pdb_file]
        pdb_path = os.path.join(pdb_folder, pdb_file)

        label_data = generate_single_pdb_label(pdb_path, rec_chain, lig_chain)
        if label_data is None:
            print(f"❌ 失败：{pdb_file}")
            fail_count += 1
            continue

        output_csv = os.path.join(output_folder, pdb_file.replace(".pdb", "_label.csv"))
        df_out = pd.DataFrame(label_data, columns=["chain_id", "residue_index", "residue_name", "interface_label", "chain_type"])
        df_out.to_csv(output_csv, index=False)
        print(f"✅ 完成：{pdb_file}")
        success_count += 1

    print(f"\n📊 {split_name} 处理完成：")
    print(f"   成功：{success_count} 个")
    print(f"   失败/跳过：{fail_count} 个")

if __name__ == "__main__":
    print("🔄 开始生成双向界面标签 (Receptor & Ligand)...")
    print(f"📂 数据集根目录：{DATASET_ROOT}")
    print(f"📂 标签输出目录：{OUTPUT_LABEL_ROOT}")
    print(f"⚙️  界面距离阈值：{DIST_THRESHOLD} 埃\n")

    process_split("train")
    process_split("val")
    process_split("test")

    print("\n🎉 全部界面标签生成完成！")
    print(f"📂 标签已保存至：{OUTPUT_LABEL_ROOT}")
    print("   ├─ train/")
    print("   ├─ val/")
    print("   └─ test/")
    print("💡 每个PDB对应一个_label.csv，包含：chain_id, residue_index, residue_name, interface_label, chain_type")