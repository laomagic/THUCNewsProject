# coding: utf-8
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=0)

        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.dropout_layer = nn.Dropout(config.dropout)

    def forward(self, x):
        embed = self.embedding(x)  # # [batch_size,seq_len,emb_size]
        out, _ = self.lstm(embed)  # [batch_size,seq_len,hidden_size*2]
        out = self.dropout_layer(out)
        out = out[:, -1, :]  # 取最后一个hidden[batch_size,hidden_size*2]
        out = self.fc(out)  # [batch_size,num_classes]

        return out
