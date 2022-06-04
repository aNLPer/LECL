import torch
import torch.nn as nn

class DescEnc(nn.Module):
    def __init__(self, embedding_size, embedding_dim):
        super(DescEnc,self).__init__()
        self.embedding = nn.Embedding(embedding_size, embedding_dim=embedding_dim)
        self.enc_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
        self.Bert = nn.TransformerEncoder(encoder_layer=self.enc_layer, num_layers=6)

    def forward(self, x):
        # [seq_length, batch_size] -> [seq_length, batch_size, d_model]
        x = self.embedding(x).view(5,10,-1)
        out = self.Bert(x)
        return out



class ArticleEnc(nn.Module):
    def __init__(self, hidden_size):
        super(ArticleEnc,self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)