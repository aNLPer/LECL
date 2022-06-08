from models.Encoder import FactEnc, AccuEnc
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataprepare.dataprepare import getAccus,Lang
import os

import pickle
import json
max_length = 0
f = open("./dataprepare/lang_data_train_preprocessed.pkl", "rb")
lang = pickle.load(f)
f.close()
# id2acc, acc2id = getAccus(os.path.join("./dataset/CAIL-SMALL","data_train_filtered.json"))
with open("./dataset/CAIL-SMALL/data_train_forModel.txt", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        print([lang.index2word[idx] for idx in item[4]])
        length = len(item[4])
        if length>max_length:
            max_length = length
print(max_length)






