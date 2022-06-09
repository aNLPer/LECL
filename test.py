from models.Encoder import FactEnc, AccuEnc
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataprepare.dataprepare import getAccus,Lang
import os


label = [[0],[0],[1],[13],[12]]
label = torch.tensor(label)
print(label)
print(list(set(label.squeeze().numpy())))






