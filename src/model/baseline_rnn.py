import itertools
import scipy as sp
import torch
import torch.nn as nn

from .common import Attention


class StackedRNNFM(nn.Module):
    def __init__(self, embedding_matrix, seq_len, hidden_size=40):
        super(StackedRNNFM, self).__init__()

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.load_state_dict({'weight': embedding_matrix})
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, seq_len)
        self.gru_attention = Attention(hidden_size * 2, seq_len)

        self.avg_pool = nn.AdaptiveAvgPool1d()
        self.max_pool = nn.AdaptiveMaxPool1d()

        fm_first_size = hidden_size * 2 * 4
        fm_second_size = hidden_size * 2 * sp.special.comb(4, 2)

        self.fc = nn.Linear(int(fm_first_size + fm_second_size), 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, inputs):
        x_embedding = self.embedding(inputs)  # B x L x D
        x_embedding = self.embedding_dropout(torch.unsqueeze(x_embedding, 0).transpose(1, 3))
        x_embedding = torch.squeeze(x_embedding.transpose(1, 3))

        x_lstm = self.lstm(x_embedding)
        x_gru = self.gru(x_lstm)

        x_lstm_attention = self.lstm_attention(x_lstm)
        x_gru_attention = self.lstm_attention(x_gru)
        x_avg_pool = self.avg_pool(x_gru)
        x_max_pool = self.max_pool(x_gru)

        fm_first = [
            x_lstm_attention,
            x_gru_attention,
            x_avg_pool,
            x_max_pool
        ]

        fm_second = []
        for t_1, t_2 in itertools.combinations(fm_first, 2):
            fm_second.append(t_1 * t_2)

        x_fc = self.fc(torch.cat(fm_first + fm_second, 1))
        x_fc = self.dropout(self.relu(x_fc))
        outputs = self.output_layer(x_fc)
        return outputs
