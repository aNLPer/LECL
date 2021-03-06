import torch
import torch.nn as nn

class FactEnc(nn.Module):
    def __init__(self, voc_size, embedding_dim):
        super(FactEnc, self).__init__()
        self.embedding = nn.Embedding(voc_size, embedding_dim=embedding_dim, padding_idx=0)
        self.dp = nn.Dropout(p=0.2)
        self.enc_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)
        self.Bert = nn.TransformerEncoder(encoder_layer=self.enc_layer, num_layers=3)
        self.linear = nn.Sequential(nn.Linear(embedding_dim, 4*embedding_dim),
                                    nn.BatchNorm1d(4*embedding_dim),
                                    nn.ReLU(),
                                    nn.Linear(4*embedding_dim, embedding_dim),
                                    nn.BatchNorm1d(embedding_dim),
                                    nn.ReLU()
                                    )

    def forward(self, x):
        # [batch_size, seq_length] -> [seq_length, batch_size]
        x = torch.transpose(x, dim0=0, dim1=1)
        # [seq_length,batch_size] -> [ seq_length, batch_size, d_model]
        x = self.embedding(x)
        x = self.dp(x)
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
        self.dp = nn.Dropout(p=0.2)
        self.hidden_size = hidden_size
        self.embedding = None
        self.gru = nn.GRU(input_size, self.hidden_size, bidirectional=True)
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

    def forward(self, x): # x [seq_length, batch_size]
        # [seq_length, batch_size] -> [seq_length, batch_size, d_model]
        x = self.embedding(x)
        x = self.dp(x)
        # [bidirectional*n_layer=2, batch_size, d_model]
        h_0 = torch.zeros(size=(2, x.shape[1], self.hidden_size)).to(self.device)
        # outputs = [seq_length, batch_size, 2*d_model]
        outputs, h_n = self.gru(x, h_0)
        # [seq_length, batch_size, 2*d_model] -> [batch_size, 2*d_model]
        outputs = self.dp(outputs)
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

    def init_weight(self):
        print("?????????factEnc......")
        for m in self.factEnc.modules():
            # ??????????????????Conv2d
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                # ?????????????????????
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zeros_()
        print("?????????accuEnc......")
        for m in self.accuEnc.modules():
            print(m)