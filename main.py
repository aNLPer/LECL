from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from timeit import default_timer as timer
from dataprepare.dataprepare import Lang,getAccus,get_acc_desc
import torch.nn as nn
import os
import torch.optim as optim
import torch
import numpy as np
import pickle
from models.Encoder import Encoder
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 24
LR_ACCU_ENC = 0.0002
LR_FACT_ENC = 0.001
SEQ_MAX_LENGTH = 500
EMBED_DIM = 256
EPOCH = 100
LABEL_DESC_MAX_LENGTH = 90 # 实际统计为83
TEMPER = 0.07
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
# 指控id-指控desc_representation
arr = [list(np.random.normal(loc=0, scale=1, size=512)) for i in range(112)]
desc_representation = np.array(arr)




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
        # label_desc = []
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
model = Encoder(voc_size=lang.n_words, embed_dim= EMBED_DIM, input_size=EMBED_DIM, hidden_size=EMBED_DIM)
model.to(device)
# 模型初始化

# 定义损失函数
def loss_fun(out_1, out_2, out_3, label_rep):
    """
    损失函数
    :param out_1: tensor
    :param out_2: tensor
    :param out_3: tensor
    :param label_rep tensor
    :return: loss scalar
    """
    batch_size = out_1.shape[0]
    # out_1 样本损失函数
    loss_out1 = 0
    for i in range(batch_size):
        # [batch_size, d_model]
        x = out_1[i].repeat(batch_size, 1)
        # [batch_size]
        x_out1 = torch.cosine_similarity(x, out_1, dim=1)/TEMPER
        # [batch_size]
        x_out2 = torch.cosine_similarity(x, out_2, dim=1)/TEMPER
        # [batch_size]
        x_out3 = torch.cosine_similarity(x, out_3, dim=1)/TEMPER
        # [batch_size]
        x_label_rep = torch.cosine_similarity(x, label_rep, dim=1)/TEMPER

        molecule = torch.sum(torch.tensor([torch.exp(x_out2[i]), torch.exp(x_out3[i]), torch.exp(x_label_rep[i])]))
        denominator = torch.sum(torch.exp(x_out1)) - torch.exp(x_out1[i]) + torch.sum(torch.exp(x_out2)) + torch.sum(torch.exp(x_out3)) + torch.sum(torch.exp(x_label_rep))
        loss_out1 -= torch.log(molecule/denominator)

    # out_2 样本损失函数
    loss_out2 = 0
    for i in range(batch_size):
        # [batch_size, d_model]
        x = out_2[i].repeat(batch_size, 1)
        # [batch_size]
        x_out1 = torch.cosine_similarity(x, out_1, dim=1) / TEMPER
        # [batch_size]
        x_out2 = torch.cosine_similarity(x, out_2, dim=1) / TEMPER
        # [batch_size]
        x_out3 = torch.cosine_similarity(x, out_3, dim=1) / TEMPER
        # [batch_size]
        x_label_rep = torch.cosine_similarity(x, label_rep, dim=1) / TEMPER

        molecule = torch.sum(torch.tensor([torch.exp(x_out1[i]), torch.exp(x_out3[i]), torch.exp(x_label_rep[i])]))
        denominator = torch.sum(torch.exp(x_out1)) + torch.sum(torch.exp(x_out2)) - torch.exp(x_out2[i]) + torch.sum(torch.exp(x_out3)) + torch.sum(torch.exp(x_label_rep))
        loss_out2 -= torch.log(molecule / denominator)

    # out_3 样本损失函数
    loss_out3 = 0
    for i in range(batch_size):
        # [batch_size, d_model]
        x = out_3[i].repeat(batch_size, 1)
        # [batch_size]
        x_out1 = torch.cosine_similarity(x, out_1, dim=1) / TEMPER
        # [batch_size]
        x_out2 = torch.cosine_similarity(x, out_2, dim=1) / TEMPER
        # [batch_size]
        x_out3 = torch.cosine_similarity(x, out_3, dim=1) / TEMPER
        # [batch_size]
        x_label_rep = torch.cosine_similarity(x, label_rep, dim=1) / TEMPER

        molecule = torch.sum(torch.tensor([torch.exp(x_out1[i]), torch.exp(x_out2[i]), torch.exp(x_label_rep[i])]))
        denominator = torch.sum(torch.exp(x_out1)) + torch.sum(torch.exp(x_out2)) + torch.sum(torch.exp(x_out3)) - torch.exp(x_out3[i]) + torch.sum(torch.exp(x_label_rep))
        loss_out3 -= torch.log(molecule / denominator)

    return loss_out1 + loss_out2 + loss_out3

# 优化器
optimizer_factEnc = optim.Adam(model.factEnc.parameters(), lr=LR_FACT_ENC)
optimizer_accuEnc = optim.Adam(model.accuEnc.parameters(), lr=LR_ACCU_ENC)


def train(epoch):
    # 设置模型为训练状态
    model.train()
    # 记录每个epoch的loss
    epoch_loss = 0
    start = timer()
    for seq_1, seq_2, seq_3, label in train_data_loader:
        # [batch_size, *] -> [batch_size, max_label_length]
        label_desc = [torch.tensor(label2desc[i.item()]) for i in label]
        label_desc = pad_sequence(label_desc)
        # 使用GPU
        seq_1, seq_2, seq_3, label_desc = seq_1.to(device), seq_2.to(device), seq_3.to(device), label_desc.to(device)
        # 梯度清零
        optimizer_factEnc.zero_grad()
        optimizer_accuEnc.zero_grad()
        # 计算模型的输出 [batch_size, d_model]
        out_1, out_2, out_3, label_rep = model(seq_1, seq_2, seq_3, label_desc)
        # 计算损失
        loss = loss_fun(out_1, out_2, out_3, label_rep)
        epoch_loss += loss.item()
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer_factEnc.step()
        optimizer_accuEnc.step()
    epoch_loss = epoch_loss/len(train_data_loader.dataset)
    end = timer()
    print(f"Epoch: {epoch},   Training Loss: {epoch_loss},  time: {(end-start)/60}s/epoch")


def evaluate():
    # 设置模型为训练状态
    # factEnc.eval()
    # accuEnc.eval()
    # 记录每个epoch的loss
    epoch_loss = 0
    # 不跟踪梯度
    with torch.no_grad():
        for seq_1, seq_2, seq_3, label_desc, label in test_data_loader:
            seq_1, seq_2, seq_3, label_desc, label = \
                seq_1.to(device), seq_2.to(device), seq_3.to(device), label_desc.to(device), label.to(device)
            # 计算模型的输出
            # out_1 = factEnc(seq_1)
            # out_2 = factEnc(seq_2)
            # out_3 = factEnc(seq_3)
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


print("start train...")
for epoch in range(50):
    train(epoch)
