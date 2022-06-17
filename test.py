from torch.nn.utils.rnn import pack_sequence,pad_sequence
import torch
import numpy as np
import torch.nn as nn

m = nn.Dropout(p=0.9)
input = torch.randn(3, 3, 3)
print(input)
output = m(input)
print(output)