from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils.commonUtils import Lang
from timeit import default_timer as timer
import torch.optim as optim
import torch
import numpy as np
import pickle
from models.Encoder import Encoder
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 24
LR_ACCU_ENC = 0.001
LR_FACT_ENC = 0.005
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

# 数据准备
seq_1, seq_2, seq_3, label_train, label2desc = prepare_training_data()
seq_1_tensor = torch.from_numpy(pad_and_cut(seq_1, SEQ_MAX_LENGTH))
seq_2_tensor = torch.from_numpy(pad_and_cut(seq_2, SEQ_MAX_LENGTH))
seq_3_tensor = torch.from_numpy(pad_and_cut(seq_3, SEQ_MAX_LENGTH))
label_tensor = torch.from_numpy(label_train)

train_data = train_dataset(seq_1_tensor, seq_2_tensor, seq_3_tensor, label_tensor)
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

seq, label_val = prepare_valid_data()
seq_val_tensor = torch.from_numpy(pad_and_cut(seq, SEQ_MAX_LENGTH))
label_val_tensor = torch.from_numpy(label_val)

val_data = val_dataset(seq_val_tensor, label_val_tensor)
val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

# 维护label-representation表
LABEL_REPRESENTATION = torch.randn(size=(len(id2acc), EMBED_DIM),dtype=torch.float32)
LABEL_REPRESENTATION.to(device)

# 实例化模型
model = Encoder(voc_size=lang.n_words, embed_dim= EMBED_DIM, input_size=EMBED_DIM, hidden_size=EMBED_DIM)
model.to(device)
# 模型初始化

# 定义损失函数
def train_loss_fun(out_1, out_2, out_3, label_rep):
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

def valid_lass_func():
    pass

def predict(outputs):
    """
    得到预测标签
    :param outputs: [batch_size, d_model]
    :return: preds
    """
    preds = []
    for rep in outputs:
        # [112, EMBED_DIM]
        reps = rep.repeat(len(LABEL_REPRESENTATION),1)
        # [112]
        similarities = torch.cosine_similarity(reps, LABEL_REPRESENTATION, dim=1)
        # max similarity corresponding index
        pred = torch.argmax(similarities)
        # batch_size
        preds.append(pred)
    return torch.tensor(preds)


# 优化器
optimizer_factEnc = optim.Adam(model.factEnc.parameters(), lr=LR_FACT_ENC)
optimizer_accuEnc = optim.Adam(model.accuEnc.parameters(), lr=LR_ACCU_ENC)

train_loss_toral = []
val_loss_total = []
def train(epoch):
    # 设置模型为训练状态
    model.train()
    # 记录每个epoch的loss
    train_loss = 0
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
        # 更新label表示向量
        for idx, val in enumerate(label):
            LABEL_REPRESENTATION[val] = label_rep[idx]
        # 计算损失
        loss = train_loss(out_1, out_2, out_3, label_rep)
        train_loss += loss.item()
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer_factEnc.step()
        optimizer_accuEnc.step()
    train_loss = train_loss/len(train_data_loader.dataset)
    train_loss_toral.append(train_loss)
    end = timer()
    print(f"Epoch: {epoch},   Training Loss: {train_loss},  time: {(end-start)/60} min/epoch")


def evaluate():
    # 设置模型为评估状态
    model.eval()
    # 记录每个epoch的loss
    val_loss = 0
    # 不跟踪梯度
    with torch.no_grad():
        for seq, label in val_data_loader:
            seq, label = seq.to(device), label.to(device)
            # 计算模型的输出 [batch_size, d_model]
            outputs = model.factEnc(seq)
            # 得到预测标签 [batch_size]
            preds = predict(outputs)
            acc = torch.sum(preds == torch.tensor(label))/len(label)
            # 计算损失
            loss = 0
            val_loss += loss.item()
    val_loss = val_loss / len(val_data_loader.dataset)
    print(f"Epoch: {epoch},   Validation Loss: {val_loss}")

print("start train...")
for epoch in range(50):
    train(epoch)
    evaluate(epoch)

