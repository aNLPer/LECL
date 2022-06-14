"""
先对accuEncoder经过几轮训练

"""

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from timeit import default_timer as timer
import torch.optim as optim
import torch
import numpy as np
import pickle
from models.Encoder import Encoder
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 24
LR_ACCU_ENC = 0.01
LR_FACT_ENC = 0.05
SEQ_MAX_LENGTH = 500
EMBED_DIM = 256
EPOCH = 100
LABEL_DESC_MAX_LENGTH = 90 # 实际统计为83
TEMPER = 1
M = 10 # distLoss的半径

# 加载语料库信息
f = open("./dataprepare/lang_data_train_preprocessed.pkl", "rb")
lang = pickle.load(f)
f.close()


f = open("./dataprepare/train_id2acc.pkl","rb")
id2acc = pickle.load(f)
f.close()

f = open("./dataprepare/train_acc2id.pkl","rb")
acc2id = pickle.load(f)
f.close()

# 指控id-指控描述idx
def getId2desc():
    accid2descidx = [[] for _ in range(112)]
    with open("./dataset/CAIL-SMALL/data_train_forModel.txt", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            id = item[3]
            desc_idx = item[4]
            accid2descidx[id] = desc_idx
    return accid2descidx

accid2descidx = getId2desc()

class train_dataset(Dataset):
    """
    训练数据集
    """
    def __init__(self, seq_1_tensor, seq_2_tensor, seq_3_tensor, label_tensor):
        self.seq_1 = seq_1_tensor
        self.seq_2 = seq_2_tensor
        self.seq_3 = seq_3_tensor
        self.label = label_tensor

    def __getitem__(self, index):
        seq_1 = self.seq_1[index]
        seq_2 = self.seq_2[index]
        seq_3 = self.seq_3[index]
        label = self.label[index]
        return seq_1, seq_2, seq_3, label

    def __len__(self):
        return len(self.seq_1)

class val_dataset(Dataset):
    """
    验证数据集
    """
    def __init__(self, seq_tensor, label_tensor):
        self.seq = seq_tensor
        self.label = label_tensor

    def __getitem__(self, index):
        return self.seq[index], self.label[index]

    def __len__(self):
        return len(self.seq)

# BERT模型输入序列填充和截取
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

def prepare_training_data():
    with open("./dataset/CAIL-SMALL/data_train_forModel.txt", "r", encoding="utf-8") as f:
        seq_1 = []
        seq_2 = []
        seq_3 = []
        label = []
        label2desc = {}
        for line in f:
            item = json.loads(line)
            seq_1.append(item[0])
            seq_2.append(item[1])
            seq_3.append(item[2])
            label.append([item[3]])
            # label_desc.append(item[4])
            if item[3] not in label2desc:
                label2desc[item[3]] = item[4]
    return np.array(seq_1), np.array(seq_2), np.array(seq_3), np.array(label), label2desc

def prepare_valid_data():
    with open("./dataset/CAIL-SMALL/data_valid_forModel(base).txt", "r", encoding="utf-8") as f:
        seq = []
        label = []
        for line in f:
            item = json.loads(line)
            seq.append(item[0])
            label.append([item[1]])
    return np.array(seq),np.array(label)