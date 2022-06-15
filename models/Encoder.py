import torch
import torch.nn as nn

class FactEnc(nn.Module):
    def __init__(self, voc_size, embedding_dim):
        super(FactEnc, self).__init__()
        self.embedding = nn.Embedding(voc_size, embedding_dim=embedding_dim, padding_idx=0)
        self.enc_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)
        self.Bert = nn.TransformerEncoder(encoder_layer=self.enc_layer, num_layers=2)
        self.linear = nn.Sequential(nn.Linear(embedding_dim, 4*embedding_dim),
                                    # nn.BatchNorm1d(4*embedding_dim),
                                    nn.ReLU(),
                                    nn.Linear(4*embedding_dim, embedding_dim),
                                    # nn.BatchNorm1d(embedding_dim),
                                    nn.ReLU()
                                    )

    def forward(self, x):
        # [batch_size, seq_length] -> [seq_length, batch_size]
        x = torch.transpose(x, dim0=0, dim1=1)
        # [seq_length,batch_size] -> [ seq_length, batch_size, d_model]
        x = self.embedding(x)
        # [ seq_length, batch_size, d_model] -> [ seq_length, batch_size, d_model]
        x = self.Bert(x)
        # [ seq_length, batch_size, d_model] -> [batch_size, d_model]
        x = torch.mean(x, dim=0)
        # [batch_size, d_model]
        out = self.linear(x)
        return out


class AccuEnc(nn.Module):
    def __init__(self, voc_size,embedding_dim,input_size, hidden_size):
        super(AccuEnc,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(voc_size, embedding_dim=embedding_dim, padding_idx=0)
        self.gru = nn.GRU(input_size, self.hidden_size, bidirectional=True)
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size * 2, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

    def forward(self, x): # x [seq_length, batch_size]
        # [seq_length, batch_size] -> [seq_length, batch_size, d_model]
        x = self.embedding(x)
        # [bidirectional*n_layer=2, batch_size, d_model]
        h_0 = torch.zeros(size=(2, x.shape[1], self.hidden_size)).to(self.device)
        # outputs = [seq_length, batch_size, 2*d_model]
        outputs, h_n = self.gru(x, h_0)
        # [seq_length, batch_size, 2*d_model] -> [batch_size, 2*d_model]
        outputs = torch.mean(outputs, dim=0)
        # [batch_size, 2*d_model] -> [batch_size, d_model]
        outputs = self.linear(outputs)
        return outputs

class Encoder(nn.Module):
    def __init__(self, voc_size, embed_dim, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.factEnc = FactEnc(voc_size, embed_dim)
        self.accuEnc = AccuEnc(voc_size, embed_dim, input_size, hidden_size)
        self.accuEnc.embedding = self.factEnc.embedding

    def forward(self, seq_1, seq_2, seq_3, label_desc):
        out_1 = self.factEnc(seq_1)
        out_2 = self.factEnc(seq_2)
        out_3 = self.factEnc(seq_3)
        label_rep = self.accuEnc(label_desc)
        return out_1, out_2, out_3, label_rep
