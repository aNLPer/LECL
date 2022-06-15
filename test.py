from torch.nn.utils.rnn import pack_sequence,pad_sequence
import torch
import numpy as np
import torch.nn as nn

b = 0
a = torch.tensor(4.3324234, dtype=torch.float32)
b += a.item()
print(b)
b += a.item()
print(b)