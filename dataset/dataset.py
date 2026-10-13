import pandas as pd
from pathlib import Path

class Datadet:
def init(self, file_path):
self.file_path = Path(file_path)
self.data = None

def load_data(self):
    # 读取数据库文件
    if not self.file_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{self.file_path}")

    suffix = self.file_path.suffix.lower()

    if suffix == ".csv":
        self.data = pd.read_csv(self.file_path)
    elif suffix in [".xlsx", ".xls"]:
        self.data = pd.read_excel(self.file_path)
    else:
        raise ValueError("不支持的文件格式")

    return self.data

def show_info(self):
    # 查看数据基本信息
    if self.data is None:
        self.load_data()

    self.data.info()
    print(self.data.head())

def get_train_data(self):
    # 获取训练集
    if self.data is None:
        self.load_data()

    return self.data[
        self.data["partition"].astype(str).str.lower().isin(
            ["train", "training", "训练集", "训练数据集"]
        )
    ]

def get_test_data(self):
    # 获取测试集
    if self.data is None:
        self.load_data()

    return self.data[
        self.data["partition"].astype(str).str.lower().isin(
            ["test", "testing", "测试集", "测试数据集"]
        )
    ]

def get_features(self):
    # 获取特征数据
    if self.data is None:
        self.load_data()

    exclude_cols = ["id", "partition", "in_train_1000", "Class"]
    return self.data.drop(columns=exclude_cols, errors="ignore")

def get_labels(self):
    # 获取类别标签
    if self.data is None:
        self.load_data()

    return self.data["Class"]

if name == "main":
dataset = Datadet("data.csv")
df = dataset.load_data()
dataset.show_info()

train_data = dataset.get_train_data()
test_data = dataset.get_test_data()