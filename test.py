from models.Encoder import FactEnc, AccuEnc
from torch.nn.utils.rnn import pack_sequence
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataprepare.dataprepare import getAccus,Lang
import os

t = torch.tensor([[1],[2],[0,0]])

print(pack_sequence(t))








