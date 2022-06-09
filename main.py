from torch.utils.data import Dataset, DataLoader
from dataprepare.dataprepare import Lang,getAccus
import torch.nn as nn
import os
import torch.optim as optim
import torch
import numpy as np
import pickle
from models.Encoder import FactEnc, AccuEnc
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
LR_ACCU_ENC = 0.0002
LR_FACT_ENC = 0.001
SEQ_MAX_LENGTH = 500
EMBED_DIM = 512
EPOCH = 100
LABEL_DESC_MAX_LENGTH = 90 # 实际统计为83

# 加载语料库信息
f = open("./dataprepare/lang_data_train_preprocessed.pkl", "rb")
lang = pickle.load(f)
f.close()
f = open("./dataprepare/train_id2acc.pkl")
id2acc = pickle.load(f)
f.close()
f = open("./dataprepare/train_acc")

class myDataset(Dataset):
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

def prepareData():
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
            if item[3] not in label2desc:
                label2desc[item[3]] = item[4]
    return np.array(seq_1), np.array(seq_2), np.array(seq_3), np.array(label), label2desc


# 数据准备
seq_1, seq_2, seq_3, label, label2desc = prepareData()
seq_1_tensor = torch.from_numpy(pad_and_cut(seq_1, SEQ_MAX_LENGTH))
seq_2_tensor = torch.from_numpy(pad_and_cut(seq_2, SEQ_MAX_LENGTH))
seq_3_tensor = torch.from_numpy(pad_and_cut(seq_3, SEQ_MAX_LENGTH))
label_tensor = torch.from_numpy(label)

train_data = myDataset(seq_1_tensor, seq_2_tensor, seq_3_tensor, label_tensor)
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = []
# 实例化模型
factEnc = FactEnc(lang.n_words, embedding_dim=EMBED_DIM)
factEnc = factEnc.to(device)
accuEnc = AccuEnc()
accuEnc = accuEnc.to(device)
# 模型初始化

# 定义损失函数

# 优化器
optimizer_factEnc = optim.Adam(factEnc.parameters(), lr=LR_FACT_ENC)
optimizer_accuEnc = optim.Adam(accuEnc.parameters(), lr=LR_ACCU_ENC)

def train(epoch):
    # 设置模型为训练状态
    factEnc.train()
    accuEnc.train()
    # 记录每个epoch的loss
    epoch_loss = 0
    for seq_1, seq_2, seq_3, label in train_data_loader:
        # 获取采样得到的labeldesc
        label_index = list(set(label.squeeze().numpy()))

        seq_1, seq_2, seq_3, label_desc, label = \
            seq_1.to(device), seq_2.to(device), seq_3.to(device), label_desc.to(device), label.to(device)
        # 梯度清零
        optimizer_factEnc.zero_grad()
        optimizer_accuEnc.zero_grad()
        # 计算模型的输出
        out_1 = factEnc(seq_1)
        out_2 = factEnc(seq_2)
        out_3 = factEnc(seq_3)
        # 计算损失
        loss = 0
        epoch_loss += loss.item()
        # 计算梯度
        loss.backward()
        # 更新梯度
        optimizer_factEnc.step()
        optimizer_accuEnc.step()
    epoch_loss = epoch_loss/len(train_data_loader.dataset)
    print(f"Epoch: {epoch},   Training Loss: {epoch_loss}")


def evaluate():
    # 设置模型为训练状态
    factEnc.eval()
    accuEnc.eval()
    # 记录每个epoch的loss
    epoch_loss = 0
    # 不跟踪梯度
    with torch.no_grad():
        for seq_1, seq_2, seq_3, label_desc, label in test_data_loader:
            seq_1, seq_2, seq_3, label_desc, label = \
                seq_1.to(device), seq_2.to(device), seq_3.to(device), label_desc.to(device), label.to(device)
            # 计算模型的输出
            out_1 = factEnc(seq_1)
            out_2 = factEnc(seq_2)
            out_3 = factEnc(seq_3)
            # 计算损失
            loss = 0
            epoch_loss += loss.item()
            # 计算梯度
            loss.backward()
            # 更新梯度
            optimizer_factEnc.step()
            optimizer_accuEnc.step()
    epoch_loss = epoch_loss / len(test_data_loader.dataset)
    print(f"Epoch: {epoch},   Training Loss: {epoch_loss}")
    print(f"Epoch: {epoch},   Validation Loss: {epoch_loss}")

if __name__=='__main__':
    print("start train...")
    for epoch in range(100):
        train(epoch)
