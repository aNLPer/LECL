from torch.utils.data import Dataset, DataLoader
from dataprepare.dataprepare import Lang
import torch
import numpy as np
import pickle
from models import Encoder
import json

import torch.nn as nn

BATCH_SIZE = 32
LR_DESC_ENC = 0.0002
LR_CASE_ENC = 0.001
SEQ_MAX_LENGTH = 500

def pad_and_cut(data, length):
    """
    填充或截二维维numpy到固定的长度
    """
    # 将2维ndarray填充和截断到固定长度
    n = len(data)
    for i in range(n):
        if len(data[i]) < length:
            # 进行填充
            data[i] = np.pad(data[i], pad_width=(0,length-len(data[i])), constant_values=0)
        if len(data[i]) > length:
            # 进行截断
            data[i] = data[i][:length]
    # 转化为np.array()形式
    new_data = np.array(data.tolist())
    return new_data

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
    return np.array(seq_1), np.array(seq_2), np.array(seq_3), np.array(label_desc), np.array(label)

class myDataset(Dataset):
    def __init__(self, seq_1_tensor, seq_2_tensor, seq_3_tensor, label_desc_tensor, label_tensor):
        self.seq_1 = seq_1_tensor
        self.seq_2 = seq_2_tensor
        self.seq_3 = seq_3_tensor
        self.label_desc = label_desc_tensor
        self.label = label_tensor

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
seq_1_tensor = torch.from_numpy(pad_and_cut(seq_1, SEQ_MAX_LENGTH))
seq_2_tensor = torch.from_numpy(pad_and_cut(seq_2, SEQ_MAX_LENGTH))
seq_3_tensor = torch.from_numpy(pad_and_cut(seq_3, SEQ_MAX_LENGTH))
label_desc_tensor = torch.from_numpy(pad_and_cut(label_desc,SEQ_MAX_LENGTH))
label_tensor = torch.from_numpy(label)

train_data = myDataset(seq_1_tensor, seq_2_tensor, seq_3_tensor, label_desc_tensor, label_tensor)
iter_train_data = DataLoader(train_data, batch_size=2,shuffle=True)
for seq_1, seq_2, seq_3, label_desc, label in iter_train_data:
    print(seq_1)
    print(torch.transpose(seq_1, dim0=0, dim1=1))
    print(seq_1)
    print(seq_2)
    print(seq_3)
    print(label_desc)
    print(label)




