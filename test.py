from torch.nn.utils.rnn import pack_sequence,pad_sequence
import torch
import numpy as np
import torch.nn as nn

a = torch.tensor([1,2,3])
print(a.expand(4,-1))



