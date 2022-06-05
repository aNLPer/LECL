from torch.utils.data import Dataset, DataLoader
import torch
from models import Encoder
import json
import torch.nn as nn

BATCH_SIZE = 32
LR_DESC_ENC = 0.0002
LR_CASE_ENC = 0.001



def prepareData():
    with open("./dataset/CAIL-SMALL/data_train_forModel.txt", "r", encoding="utf-8") as f:
        seq_1 = []
        seq_2 = []
        seq_3 = []
        label = []
        label_desc = []
        for line in f:
            item = json.loads(line)
            seq_1.append(item[0])
            seq_2.append(item[1])
            seq_3.append(item[2])
            label.append([item[3]])
            label_desc.append(item[4])
    return seq_1, seq_2, seq_3, label_desc, label


class myDataset(Dataset):
    def __init__(self, seq_1, seq_2, seq_3, label_desc, label):
        self.seq_1 = torch.tensor(seq_1, dtype=torch.long)
        self.seq_2 = torch.tensor(seq_2, dtype=torch.long)
        self.seq_3 = torch.tensor(seq_3, dtype=torch.long)
        self.label_desc = torch.tensor(label_desc, dtype=torch.long)
        self.label = torch.tensor(label, dtype=torch.long)

    def __getitem__(self, index):
        seq_1 = self.seq_1[index]
        seq_2 = self.seq_2[index]
        seq_3 = self.seq_3[index]
        label_desc = self.label_desc[index]
        label = self.label[index]
        return seq_1, seq_2, seq_3, label_desc, label

    def __len__(self):
        return len(self.seq_1)

seq_1, seq_2, seq_3, label_desc, label = prepareData()
train_data = myDataset(seq_1, seq_2, seq_3,label_desc, label)
iter_train_data = DataLoader(train_data, batch_size=2,shuffle=True)
for seq_1, seq_2, seq_3, label_desc, label in iter_train_data:
    print(seq_1)
    print(seq_2)
    print(seq_3)
    print(label_desc)
    print(label)