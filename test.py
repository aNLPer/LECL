from models.Encoder import FactEnc, AccuEnc
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

import torch

sample1 = torch.ones(5,10)  #第一个序列的长度为5
sample2 = 2*torch.ones(4,10)  #第二个序列的长度为4
sample3 = 3*torch.ones(3,10)  #第三个序列的长度为3

sequence = torch.nn.utils.rnn.pack_sequence([sample1, sample2, sample3])
print(sequence)
print(sequence.data.size()) # torch.Size([12, 10])
print(sequence.batch_sizes) # tensor([3, 3, 3, 2, 1])
for t in sequence:
    print(t)







