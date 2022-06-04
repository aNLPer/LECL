from torch.utils.data import Dataset, DataLoader
import torch
from models import Encoder
import torch.nn as nn


class NormalDataset(Dataset):
    def __init__(self, seq_tensor, label_tensor):
        self.seq_tensor = seq_tensor
        self.label_tensor = label_tensor

    def __getitem__(self, index):
        seq = self.seq_tensor[index]
        label = int(self.label_tensor[index])
        return seq, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.seq_tensor)


encoder = Encoder.BertEnc(20, 8)
x = torch.randint(0,20,(5, 10))
end = encoder(x)
print(end)