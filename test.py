import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

seq_tensor_1 = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
seq_tensor_2 = [[1,2,3],[5,6],[9,10,11],[13,14,15,16,1,1,1]]
seq_label = [0,1,0,1]

def pad_and_cut(data, length):
    """填充或截二维维numpy到固定的长度"""
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

# seq = torch.tensor(pad_and_cut(np.array(seq_tensor_2),5)).float()
seq = torch.tensor(seq_tensor_1)
seq = torch.transpose(seq, dim0=0, dim1=1)
print(seq)







class myDataset(Dataset):
    def __init__(self, seq_1, seq_2, label):
        super(myDataset, self).__init__()
        self.seq_tensor_1 = torch.tensor(seq_1, dtype=torch.long)
        self.seq_tensor_2 = torch.tensor(seq_2, dtype=torch.long)
        self.label_tensor = torch.tensor(label, dtype=torch.long)
    def __getitem__(self, item):
        return self.seq_tensor_1[item], self.seq_tensor_2[item], self.label_tensor[item]

    def __len__(self):
        return len(self.seq_tensor_1)

# dataset = myDataset(seq_tensor_1,seq_tensor_2, seq_label)
#
# train_data = DataLoader(dataset, batch_size=2, shuffle=True)
# for data_1,data_2,  label in train_data:
#     print(data_1)
#     print(data_1[0])
#     print(data_2)
#     print(label)