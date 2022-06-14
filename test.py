from torch.nn.utils.rnn import pack_sequence,pad_sequence
import torch
import numpy as np
import torch.nn as nn

a = torch.randn(2, 2, requires_grad=True) # 缺失情况下默认 requires_grad = False
a = ((a * 3) / (a - 1))

b = (a * a).sum()
print(b.requires_grad)
print(b.grad_fn)




