from torch.nn.utils.rnn import pack_sequence,pad_sequence
import torch
import numpy as np
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
a = torch.ones(25)
b = torch.ones(22)+1
c = torch.ones(15)+2
end = pad_sequence([a, b, c])
print(end.size())











