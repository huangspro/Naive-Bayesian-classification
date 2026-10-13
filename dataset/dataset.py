import pandas as pd
from pathlib import Path

class BeanDataset:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.data = []
        self.train_set = []
        self.val_set = []
        self.load_data()
        
    def load_data(self):
        print("dataset loading")
        f = pd.read_csv(self.file_path)
        rows = f.to_dict(orient='records')
        self.data = rows
        if "partition" in self.data[0]:
            self.train_set = [index for index, i in enumerate(self.data) if i["partition"] == "TRAIN"]
            self.val_set = [index for index, i in enumerate(self.data) if i["partition"] == "TEST"]
        else:
            self.train_set = [i for i in range(0, len(self.data))]
            
    def length(self):
        return (len(self.data), len(self.train_set), len(self.val_set))
        
    def get(self, idx):
        input_features = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity", "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"]
        output_features = "Class"
        class_name = ["SEKER", "BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SIRA"]
        inputs = []
        outputs = -1
        for i in input_features:
            for k,v in self.data[idx].items():
                if k == i:
                    inputs.append(v)
                    break
        outputs = self.data[idx][output_features]
        return (inputs, outputs)
        
    def get_in_train(self, idx):
        if self.train_set != None:
            return self.get(self.train_set[idx])
        else:
            return None
            
    def get_in_val(self, idx):
        if self.val_set != None:
            return self.get(self.val_set[idx])
        else:
            return None

