from torch.nn.utils.rnn import pack_sequence,pad_sequence
import torch
import numpy as np
import torch.nn as nn

x = torch.tensor([[2,4,3],
                  [2,3,4]], dtype=torch.float32)

t1 = torch.tensor([[1,2,3],
                   [1,2,3]], dtype=torch.float32)
print((x-t1))
print((x-t1)**2)
print(torch.sum(0.5*(x-t1)**2,dim=1))