from torch.nn.utils.rnn import pack_sequence,pad_sequence
import torch
import numpy as np
import torch.nn as nn

t = 0
x = torch.tensor(10,dtype=torch.float32, requires_grad=True)
for i in range(3):
    t = t+x
print(t)