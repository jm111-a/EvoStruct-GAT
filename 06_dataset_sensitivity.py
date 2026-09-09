import os
import torch
import warnings

from torch_geometric.data import Dataset


warnings.filterwarnings(
    'ignore'
)



# =====================================================
# 路径配置
# =====================================================


ROOT = (
    r"C:\Users\Administrator\Desktop\P-P"
)


# =====================================================
# 当前 sensitivity 实验
#
# 可选：
#
# "4A"
# "4p5A"
# "deltaSASA1"
#
# 每次训练前只修改这里
# =====================================================

EXPERIMENT = "4p5A"


VALID_EXPERIMENTS = [
    "4A",
    "4p5A",
    "deltaSASA1"
]


if EXPERIMENT not in VALID_EXPERIMENTS:

    raise ValueError(
        f"未知EXPERIMENT: {EXPERIMENT}，"
        f"可选值为: {VALID_EXPERIMENTS}"
    )



# =====================================================
# sensitivity graph根目录
# =====================================================


GRAPH_ROOT = os.path.join(
    ROOT,
    "cluster_split_v3_A2",
    "sensitivity_analysis",
    EXPERIMENT,
    "06_graphs"
)



# graph目录

PYG_GRAPH_TRAIN = os.path.join(
    GRAPH_ROOT,
    "train_pyg_3d_graph"
)


PYG_GRAPH_VAL = os.path.join(
    GRAPH_ROOT,
    "val_pyg_3d_graph"
)


PYG_GRAPH_TEST = os.path.join(
    GRAPH_ROOT,
    "test_pyg_3d_graph"
)



# =====================================================
# 模型保存
#
# 三套 sensitivity 实验分别保存
# 防止 best_model.pth 互相覆盖
# =====================================================


MODEL_SAVE_DIR = os.path.join(
    ROOT,
    "cluster_split_v3_A2",
    "sensitivity_analysis",
    EXPERIMENT,
    "07_model_save"
)


os.makedirs(
    MODEL_SAVE_DIR,
    exist_ok=True
)




# =====================================================
# Dataset
# =====================================================


class PPIDataset(Dataset):


    def __init__(
            self,
            pyg_dir,
            mode="train"
    ):


        super().__init__(
            None,
            None,
            None
        )


        self.pyg_dir = pyg_dir

        self.mode = mode



        if os.path.exists(
            pyg_dir
        ):


            self.file_list = [

                f

                for f in os.listdir(
                    pyg_dir
                )

                if f.endswith(
                    ".pyg"
                )

            ]


        else:


            self.file_list = []



        self.file_list.sort()



        if len(self.file_list)==0:

            print(
                f"⚠️ {mode} 集没有找到pyg文件:"
            )

            print(
                pyg_dir
            )




    def len(self):

        return len(
            self.file_list
        )




    def get(
            self,
            idx
    ):


        file_name = (
            self.file_list[idx]
        )


        pyg_path = os.path.join(
            self.pyg_dir,
            file_name
        )



        # PyTorch 2.6兼容

        try:

            data = torch.load(
                pyg_path,
                weights_only=False
            )


        except Exception:


            data = torch.load(
                pyg_path
            )



        # 标签检查

        if (
            not hasattr(
                data,
                "y"
            )

            or

            data.y is None

        ):


            raise ValueError(
                f"{file_name} 缺少y标签"
            )



        return data





# =====================================================
# 加载
# =====================================================


print("="*70)

print(
    "Loading EvoStruct-GAT sensitivity graph dataset"
)


print(
    "Experiment:",
    EXPERIMENT
)


print(
    "Train:",
    PYG_GRAPH_TRAIN
)

print(
    "Val:",
    PYG_GRAPH_VAL
)

print(
    "Test:",
    PYG_GRAPH_TEST
)



train_dataset = PPIDataset(
    PYG_GRAPH_TRAIN,
    mode="train"
)


val_dataset = PPIDataset(
    PYG_GRAPH_VAL,
    mode="val"
)


test_dataset = PPIDataset(
    PYG_GRAPH_TEST,
    mode="test"
)




print("\nDataset size:")

print(
    "Train graphs:",
    len(train_dataset)
)

print(
    "Val graphs:",
    len(val_dataset)
)

print(
    "Test graphs:",
    len(test_dataset)
)


print(
    "Model save dir:",
    MODEL_SAVE_DIR
)




# =====================================================
# 测试样本
# =====================================================


if __name__=="__main__":


    if len(train_dataset)>0:


        data = train_dataset[0]


        print(
            "\nFirst graph:"
        )


        print(
            "file:",
            train_dataset.file_list[0]
        )


        print(
            "x:",
            data.x.shape
        )


        print(
            "edge:",
            data.edge_index.shape
        )


        print(
            "edge_attr:",
            data.edge_attr.shape
        )


        print(
            "y:",
            data.y.shape
        )


        print(
            "interface ratio:",
            f"{100*data.y.sum().item()/data.y.shape[0]:.2f}%"
        )


        print(
            "\n✅ Dataset check passed"
        )


    else:


        print(
            "\n❌ 没找到graph文件"
        )



print("="*70)
