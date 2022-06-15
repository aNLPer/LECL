from torch.nn.utils.rnn import pack_sequence,pad_sequence
import torch
import numpy as np
import torch.nn as nn

a = torch.tensor([1,2,3],dtype=torch.float32)
print(a.long())
print(a)