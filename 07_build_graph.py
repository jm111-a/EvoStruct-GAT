import os
import random
import torch
import esm
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.SeqUtils import seq1
from torch_geometric.data import Data
import warnings

warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(f"设备: {DEVICE}")

METADATA_CSV = r"C:\Users\Administrator\Desktop\P-P\filtered_pdb_metadata_final.csv"
DATASET_ROOT = r"C:\Users\Administrator\Desktop\P-P\dataset_split"
LABEL_DIR = r"C:\Users\Administrator\Desktop\P-P\labels"
OUTPUT_BASE = r"C:\Users\Administrator\Desktop\P-P\all_output"

DSSP_EXE_PATH = r"C:\Users\Administrator\Desktop\P-P\dssp.exe"


def set_all_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_all_seed(42)

print(f"🚀 正在加载 esm2_t33_650M_UR50D 到 {DEVICE}...")
model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model_esm = model_esm.to(DEVICE).eval()
batch_converter = alphabet.get_batch_converter()


def process_chain_graph(pdb_path, chain_ids_str, labels_df, out_dirs, chain_type, pdb_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    model = structure[0]

    target_chain_ids = list(str(chain_ids_str))
    all_residues = []

    for c_id in target_chain_ids:
        if c_id in model:
            for res in model[c_id]:
                if res.id[0] == ' ':
                    all_residues.append(res)

    if not all_residues:
        return None

    dssp_dict = None
    try:
        dssp_dict = dict(DSSP(model, pdb_path, dssp=DSSP_EXE_PATH))
    except Exception as e:
        print(f"\n⚠️ 物理版 dssp.exe 计算失败 ({pdb_id}): {e}")

    seq = ""
    for res in all_residues:
        aa = seq1(res.get_resname(), custom_map={"MSE": "M"})
        seq += aa if aa not in ['', '?'] else 'X'

    if len(seq) > 1022: seq = seq[:1022]

    data_esm = [(pdb_id, seq)]
    _, _, batch_tokens = batch_converter(data_esm)
    batch_tokens = batch_tokens.to(DEVICE)

    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33][0, 1: len(seq) + 1].cpu().numpy()

    relevant_labels = labels_df[labels_df['chain_id'].isin(target_chain_ids)]

    node_features = []
    node_coords = []
    node_labels = []
    csv_rows = []

    valid_count = min(len(all_residues), len(token_representations))
    for i in range(valid_count):
        res = all_residues[i]
        res_idx = res.get_id()[1]
        c_id = res.get_full_id()[2]

        match = relevant_labels[(relevant_labels['residue_index'] == res_idx) &
                                (relevant_labels['chain_id'] == c_id)]
        if match.empty: continue

        y_val = match['interface_label'].values[0]

        struct_feat = [0.0, 0.0, 1.0, 0.5]
        sec_struct_char = '-'

        if dssp_dict is not None:
            dssp_key = (c_id, res.get_id())
            if dssp_key in dssp_dict:
                dssp_val = dssp_dict[dssp_key]
                sec_struct_char = dssp_val[2]
                rsa = dssp_val[3]

                is_H = 1.0 if sec_struct_char in ['H', 'G', 'I'] else 0.0
                is_E = 1.0 if sec_struct_char in ['B', 'E'] else 0.0
                is_C = 1.0 if sec_struct_char in ['T', 'S', '-'] else 0.0

                try:
                    rsa = float(rsa)
                except (ValueError, TypeError):
                    rsa = 0.5

                struct_feat = [is_H, is_E, is_C, rsa]

        full_feat = np.concatenate([token_representations[i], struct_feat])
        coord = res['CA'].get_coord() if 'CA' in res else res.child_list[0].get_coord()

        node_features.append(full_feat)
        node_coords.append(coord)
        node_labels.append(y_val)

        csv_rows.append({
            "PDB_ID": pdb_id, "Chain": c_id, "Res_Name": res.get_resname(), "Res_ID": res_idx,
            "Coord_X": round(coord[0], 2), "Coord_Y": round(coord[1], 2), "Coord_Z": round(coord[2], 2),
            "Sec_Struct": sec_struct_char, "Is_Helix": struct_feat[0], "Is_Sheet": struct_feat[1],
            "Is_Coil": struct_feat[2], "RSA": round(struct_feat[3], 4), "Label": y_val
        })

    if len(node_features) < 5: return None

    x = torch.tensor(np.array(node_features), dtype=torch.float)
    pos = torch.tensor(np.array(node_coords), dtype=torch.float)
    y = torch.tensor(np.array(node_labels), dtype=torch.long)

    dist_matrix = torch.cdist(pos, pos)
    edge_index = (dist_matrix < 10.0).nonzero(as_tuple=False).t()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]

    dist_val = dist_matrix[edge_index[0], edge_index[1]]
    edge_attr = torch.exp(
        -0.5 * (dist_val.unsqueeze(-1) / torch.tensor([1, 2, 5, 10, 20], device=dist_val.device)) ** 2)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)

    node_degrees = torch.bincount(edge_index[0], minlength=len(node_features)).cpu().numpy()
    for i in range(len(csv_rows)):
        csv_rows[i]["Node_Degree_10A"] = node_degrees[i]

    df_features = pd.DataFrame(csv_rows)
    cols_order = ["PDB_ID", "Chain", "Res_Name", "Res_ID", "Label", "Node_Degree_10A",
                  "Sec_Struct", "Is_Helix", "Is_Sheet", "Is_Coil", "RSA", "Coord_X", "Coord_Y", "Coord_Z"]
    df_features = df_features[cols_order]

    df_features.to_csv(os.path.join(out_dirs['csv'], f"{pdb_id}_{chain_type}_features.csv"), index=False)
    torch.save(data, os.path.join(out_dirs['pyg'], f"{pdb_id}_{chain_type}.pyg"))
    return True


def main():
    df_meta = pd.read_csv(METADATA_CSV)
    for mode in ["train", "val", "test"]:
        print(f"\n📂 正在处理 {mode} 分区...")
        pdb_dir = os.path.join(DATASET_ROOT, mode)
        if not os.path.exists(pdb_dir): continue

        out_dirs = {
            'pyg': os.path.join(OUTPUT_BASE, f"{mode}_pyg_3d_graph"),
            'csv': os.path.join(OUTPUT_BASE, f"{mode}_csv_features")
        }
        for d in out_dirs.values(): os.makedirs(d, exist_ok=True)

        files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
        for f in tqdm(files):
            pdb_id = f.replace(".pdb", "")
            pdb_path = os.path.join(pdb_dir, f)
            label_path = os.path.join(LABEL_DIR, mode, f"{pdb_id}_label.csv")

            if not os.path.exists(label_path): continue
            labels_df = pd.read_csv(label_path)

            meta_row = df_meta[df_meta['pdb_file'] == f]
            if meta_row.empty: continue

            rec_chain_ids = str(meta_row['receptor_chain'].values[0])
            lig_chain_ids = str(meta_row['ligand_chain'].values[0])

            try:
                process_chain_graph(pdb_path, rec_chain_ids, labels_df, out_dirs, "rec", pdb_id)
                process_chain_graph(pdb_path, lig_chain_ids, labels_df, out_dirs, "lig", pdb_id)
            except Exception:
                pass


if __name__ == "__main__":
    main()