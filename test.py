from torch.nn.utils.rnn import pack_sequence,pad_sequence
import torch
import numpy as np
import torch.nn as nn
a = torch.tensor([[1,3,4],[1,2,3]],dtype=torch.float32)
b = torch.tensor([[1,2,4],[1,2,5]],dtype=torch.float32)
cosim = torch.cosine_similarity(a,b, dim=1)
print(cosim)
print(cosim/2)










