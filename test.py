from torch.nn.utils.rnn import pack_sequence,pad_sequence
import torch
import numpy as np
import torch.nn as nn

m = nn.LeakyReLU()
input = torch.tensor([0.01, -0.3])
print(input)
output = m(input)
print(output)

print(input.shape)