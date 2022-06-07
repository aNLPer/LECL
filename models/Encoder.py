import torch
import torch.nn as nn

class FactEnc(nn.Module):
    def __init__(self, voc_size, embedding_dim):
        super(FactEnc, self).__init__()
        self.embedding = nn.Embedding(voc_size, embedding_dim=embedding_dim)
        self.enc_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
        self.Bert = nn.TransformerEncoder(encoder_layer=self.enc_layer, num_layers=6)

    def forward(self, x):
        # [seq_length, batch_size] -> [batch_size, seq_length]
        x = torch.transpose(x, dim0=0, dim1=1)
        # [batch_size, seq_length] -> [batch_size, seq_length, d_model]
        x = self.embedding(x)
        out = self.Bert(x)
        return out



class AccuEnc(nn.Module):
    def __init__(self, hidden_size):
        super(AccuEnc,self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)