import torch
import torch.nn as nn

class FactEnc(nn.Module):
    def __init__(self, voc_size, embedding_dim):
        super(FactEnc, self).__init__()
        self.embedding = nn.Embedding(voc_size, embedding_dim=embedding_dim)
        self.enc_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
        self.Bert = nn.TransformerEncoder(encoder_layer=self.enc_layer, num_layers=6)
        self.linear = nn.Sequential(nn.Linear(512, 2048),
                                    nn.BatchNorm1d(2048),
                                    nn.ReLU(),
                                    nn.Linear(2048, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU()
                                    )
    def forward(self, x):
        # [batch_size, seq_length] -> [seq_length, batch_size]
        x = torch.transpose(x, dim0=0, dim1=1)
        # [seq_length,batch_size] -> [ seq_length, batch_size, d_model]
        x = self.embedding(x)
        # [ seq_length, batch_size, d_model]
        x = self.Bert(x)
        # [batch_size, d_model]
        x = torch.sum(x, dim=0)
        # [batch_size, d_model]
        out = self.linear(x)
        return out



class AccuEnc(nn.Module):
    def __init__(self, hidden_size):
        super(AccuEnc,self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, x):

        return x