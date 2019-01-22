import itertools
import scipy as sp
import torch
import torch.nn as nn

from .common import GeneralAttention


class AttentionMaskRNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=64, out_hidden_dim=32, attention_type='general',
                 embed_drop=0.1, out_drop=0.2, mask=False):
        super(AttentionMaskRNN, self).__init__()

        self.mask = mask

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(embed_drop)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention, _ = GeneralAttention(hidden_size * 2, attention_type)
        self.gru_attention, _ = GeneralAttention(hidden_size * 2, attention_type)

        fm_first_size = hidden_size * 2 * 4
        fm_second_size = hidden_size * 2 * sp.special.comb(4, 2)

        self.fc = nn.Linear(int(fm_first_size + fm_second_size), out_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(out_drop)
        self.output_layer = nn.Linear(out_hidden_dim, 1)

    def forward(self, inputs):
        if self.mask:
            x_embedding = self.embedding(inputs['sequence'])  # B x L x D
        else:
            x_embedding = self.embedding(inputs)  # B x L x D
        x_embedding = self.embedding_dropout(torch.unsqueeze(x_embedding, 0).transpose(1, 3))
        x_embedding = torch.squeeze(x_embedding.transpose(1, 3))

        x_lstm, _ = self.lstm(x_embedding)
        x_gru, _ = self.gru(x_lstm)

        x_lstm_attention = self.lstm_attention(x_lstm, x_lstm)
        x_gru_attention = self.gru_attention(x_gru, x_gru)

        if self.mask:
            x_lstm_attention = x_lstm_attention * inputs['mask'].unsqueeze(-1)
            x_gru_attention = x_gru_attention * inputs['mask'].unsqueeze(-1)

        x_avg_pool_lstm = torch.mean(x_lstm_attention, 1)
        x_max_pool_lstm, _ = torch.max(x_lstm_attention, 1)
        x_avg_pool_gru = torch.mean(x_gru_attention, 1)
        x_max_pool_gru, _ = torch.max(x_gru_attention, 1)

        fm_first = [
            x_avg_pool_lstm,
            x_max_pool_lstm,
            x_avg_pool_gru,
            x_max_pool_lstm
        ]

        fm_second = []
        for t_1, t_2 in itertools.combinations(fm_first, 2):
            fm_second.append(t_1 * t_2)

        x_fc = self.fc(torch.cat(fm_first + fm_second, 1))
        x_fc = self.dropout(self.relu(x_fc))
        outputs = self.output_layer(x_fc)
        return outputs