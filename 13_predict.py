import os
import torch
import esm
import numpy as np
import pandas as pd
import importlib
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.SeqUtils import seq1
from torch_geometric.data import Data
import warnings
import ctypes

if os.name == 'nt':
    ctypes.windll.kernel32.SetErrorMode(0x0002 | 0x8000)

warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = r"C:\Users\Administrator\Desktop\P-P\model_save\best_model.pth"
DSSP_EXE_PATH = r"C:\Users\Administrator\Desktop\P-P\dssp.exe"

print("=" * 60)
print("=" * 60)
print(f"🔥 当前计算设备: {DEVICE}")

try:
    model_module = importlib.import_module("09_model")
    PPI_Model = getattr(model_module, "EvoStruct_GAT", None)
    if PPI_Model is None:
        raise ImportError("未在 09_model.py 中找到有效的模型类")
except Exception as e:
    print(f"❌ 模型模块加载失败，报错: {e}")
    exit()


def build_graph_for_prediction(pdb_path, target_chain, model_esm, batch_converter):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", pdb_path)
    model_struct = structure[0]

    all_residues = []
    res_mapping_info = []

    if target_chain in model_struct:
        for res in model_struct[target_chain]:
            if res.id[0] == ' ':
                all_residues.append(res)
    else:
        raise ValueError(f"❌ PDB 文件中未找到链 '{target_chain}'！")

    if not all_residues:
        raise ValueError(f"❌ 链 '{target_chain}' 中没有检测到有效的氨基酸！")

    dssp_dict = None
    try:
        dssp_dict = dict(DSSP(model_struct, pdb_path, dssp=DSSP_EXE_PATH))
    except Exception as e:
        print(f"⚠️ 物理引擎 dssp.exe 计算失败，将使用默认 Coil 结构兜底: {e}")

    seq = ""
    for res in all_residues:
        aa = seq1(res.get_resname(), custom_map={"MSE": "M"})
        seq += aa if aa not in ['', '?'] else 'X'

    if len(seq) > 1022:
        print(f"⚠️ 警告: 链 {target_chain} 长度超过 1022，超出部分将被截断！")
        seq = seq[:1022]

    data_esm = [("target", seq)]
    _, _, batch_tokens = batch_converter(data_esm)
    batch_tokens = batch_tokens.to(DEVICE)

    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33][0, 1: len(seq) + 1].cpu().numpy()

    node_features = []
    node_coords = []

    valid_count = min(len(all_residues), len(token_representations))
    for i in range(valid_count):
        res = all_residues[i]
        c_id = res.get_full_id()[2]
        res_idx = res.get_id()[1]
        res_name = res.get_resname()
        res_mapping_info.append({"chain": c_id, "res_id": res_idx, "res_name": res_name})

        struct_feat = [0.0, 0.0, 1.0, 0.5]  # 默认 Coil, RSA=0.5

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

        # 拼接 1280 (ESM) + 4 (DSSP) = 1284
        full_feat = np.concatenate([token_representations[i], struct_feat])

        # 获取坐标
        coord = res['CA'].get_coord() if 'CA' in res else res.child_list[0].get_coord()

        node_features.append(full_feat)
        node_coords.append(coord)

    x = torch.tensor(np.array(node_features), dtype=torch.float)
    pos = torch.tensor(np.array(node_coords), dtype=torch.float)

    dist_matrix = torch.cdist(pos, pos)
    edge_index = (dist_matrix < 10.0).nonzero(as_tuple=False).t()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]

    dist_val = dist_matrix[edge_index[0], edge_index[1]]
    edge_attr = torch.exp(
        -0.5 * (dist_val.unsqueeze(-1) / torch.tensor([1, 2, 5, 10, 20], device=dist_val.device)) ** 2
    )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    return data, res_mapping_info


def run_prediction(pdb_file, target_chain, threshold):
    print(f"\n🚀 正在唤醒 ESM-2 语言大模型 ...")
    model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model_esm = model_esm.to(DEVICE).eval()
    batch_converter = alphabet.get_batch_converter()

    print(f"⚙️ 正在装载 GNN 最优权重: {os.path.basename(WEIGHTS_PATH)} ...")
    model = PPI_Model(in_dim=1284, edge_dim=5).to(DEVICE)
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"❌ 权重加载失败！报错: {e}")
        return
    model.eval()

    all_predictions = []

    print(f"\n🧬 正在处理链 [{target_chain}] (提取特征、物理建图)...")
    try:
        graph_data, res_info = build_graph_for_prediction(pdb_file, target_chain, model_esm, batch_converter)
        graph_data = graph_data.to(DEVICE)

        # 模型前向推理
        with torch.no_grad():
            logits = model(graph_data)
            probs = torch.sigmoid(logits).cpu().numpy()

        # 解析结果
        for i, info in enumerate(res_info):
            p = probs[i]
            is_hotspot = 1 if p >= threshold else 0
            all_predictions.append({
                "Chain": info["chain"],
                "Res_Name": info["res_name"],
                "PDB_ID": info["res_id"],
                "Probability": round(float(p), 4),
                "Prediction": "Hotspot" if is_hotspot else "Non-interface"
            })
    except Exception as e:
        print(f"❌ 处理中断: {e}")
        return

    if not all_predictions:
        return

    df_results = pd.DataFrame(all_predictions)
    hotspots_df = df_results[df_results["Prediction"] == "Hotspot"].sort_values(by="Probability", ascending=False)

    print("\n" + "=" * 60)
    print(f"🎯 预测完成！(执行严格阈值: {threshold})")
    print(f"📊 链 {target_chain} 总计扫描氨基酸: {len(df_results)} 个 | 发现潜在相互作用热点: {len(hotspots_df)} 个")
    print("=" * 60)

    if not hotspots_df.empty:
        print("🏆 【高置信度结合热点列表 (按可能性降序)】")
        print(f"{'链':<4} | {'残基名称':<8} | {'PDB编号':<8} | {'预测概率':<10}")
        print("-" * 45)
        # 屏幕最多打印 Top 20
        for _, row in hotspots_df.head(20).iterrows():
            print(f"{row['Chain']:<5} | {row['Res_Name']:<10} | {row['PDB_ID']:<10} | {row['Probability']:.4f}")

        if len(hotspots_df) > 20:
            print(f"... 还有 {len(hotspots_df) - 20} 个热点未显示，请查看完整 CSV 文件。")
    else:
        print("⚠️ 模型认为该链表面没有超过设定阈值的明显结合位点。")

    # 保存为 CSV
    base_name = os.path.basename(pdb_file).replace(".pdb", "")
    out_csv = f"{base_name}_chain_{target_chain}_predictions.csv"

    df_results = df_results[["Chain", "Res_Name", "PDB_ID", "Probability", "Prediction"]]
    df_results.to_csv(out_csv, index=False)
    print(f"\n💾 完整打分报告已保存至: {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    while True:
        pdb_input = input("\n👉 1. 请输入待预测的 PDB 文件路径 (如 data/test.pdb): ").strip()
        if os.path.exists(pdb_input):
            break
        print("❌ 找不到该文件，请检查路径是否正确！")

    target_chain = input("👉 2. 请输入你想预测的目标链 ID (单链输入，如 A): ").strip()

    thresh_input = input("👉 3. 请输入分类阈值 (直接回车将使用模型最优推荐值 0.59): ").strip()
    threshold = float(thresh_input) if thresh_input else 0.59

    # 启动预测
    run_prediction(pdb_input, target_chain, threshold)