import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

seq_tensor_1 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
seq_tensor_2 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
seq_label = [0,1,0,1]

class myDataset(Dataset):
    def __init__(self, seq_1, seq_2, label):
        super(myDataset, self).__init__()
        self.seq_tensor_1 = torch.tensor(seq_1, dtype=torch.long)
        self.seq_tensor_2 = torch.tensor(seq_2, dtype=torch.long)
        self.label_tensor = torch.tensor(label, dtype=torch.long)
    def __getitem__(self, item):
        return self.seq_tensor_1[item], self.seq_tensor_2[item], self.label_tensor[item]

    def __len__(self):
        return len(self.seq_tensor_1)

dataset = myDataset(seq_tensor_1,seq_tensor_2, seq_label)

train_data = DataLoader(dataset, batch_size=2, shuffle=True)
for data_1,data_2,  label in train_data:
    print(data_1)
    print(data_2)
    print(label)