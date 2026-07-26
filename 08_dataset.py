import os
import torch
import warnings
from torch_geometric.data import Dataset

warnings.filterwarnings('ignore')

PYG_GRAPH_TRAIN = r"C:\Users\Administrator\Desktop\P-P\all_output\train_pyg_3d_graph"
PYG_GRAPH_VAL = r"C:\Users\Administrator\Desktop\P-P\all_output\val_pyg_3d_graph"
PYG_GRAPH_TEST = r"C:\Users\Administrator\Desktop\P-P\all_output\test_pyg_3d_graph"

MODEL_SAVE_DIR = r"C:\Users\Administrator\Desktop\P-P\model_save"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


class PPIDataset(Dataset):
    def __init__(self, pyg_dir, mode='train'):
        super().__init__(None, None, None)
        self.pyg_dir = pyg_dir
        self.mode = mode

        if os.path.exists(pyg_dir):
            self.file_list = [f for f in os.listdir(pyg_dir) if f.endswith('.pyg')]
        else:
            self.file_list = []

        if len(self.file_list) == 0:
            print(f"⚠️ 警告: {mode} 集目录不存在或没有 .pyg 文件")

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        file_name = self.file_list[idx]
        pyg_path = os.path.join(self.pyg_dir, file_name)

        try:
            data = torch.load(pyg_path, weights_only=False)
        except Exception:
            data = torch.load(pyg_path)

        if not hasattr(data, 'y') or data.y is None:
            raise ValueError(f"❌ 错误: 文件 {file_name} 中不包含标签 'y'！请重新检查")

        return data


print("=" * 60)
print("✅ 正在从 .pyg 文件直接加载图数据与标签...")

train_dataset = PPIDataset(PYG_GRAPH_TRAIN, mode='train')
val_dataset = PPIDataset(PYG_GRAPH_VAL, mode='val')
test_dataset = PPIDataset(PYG_GRAPH_TEST, mode='test')

if __name__ == "__main__":
    if len(train_dataset) > 0:
        data = train_dataset[0]
        print(f"\n📊 预览第一个样本 ({train_dataset.file_list[0]}):")
        print(f"   - 节点特征 (x): {data.x.shape}")
        print(f"   - 界面标签 (y): {data.y.shape}")
        print(f"   - 边信息 (edge): {data.edge_index.shape[1]} 条边")
        print(f"   - 界面残基比例: {100 * data.y.sum().item() / data.y.shape[0]:.2f}%")
    else:
        print("\n⚠️ 未找到有效数据，请检查all_output 文件夹路径。")
print("=" * 60)