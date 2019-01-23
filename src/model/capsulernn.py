import itertools
import scipy as sp
import torch
import torch.nn as nn

from .common import Attention, Dense
from .capsule import CapsuleUnit
from .transformer import get_non_pad_mask


class CapsuleRNN(nn.Module):
    def __init__(self, embedding_matrix, seq_length, hidden_dim=64, out_hidden_dim=64, out_drop=0.3, embed_drop=0.2,
                 num_capsule=16, dim_capsule=16, routings=4):
        super(CapsuleRNN, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        self.embedding_dropout = nn.Dropout2d(embed_drop)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_dim, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.capsule = CapsuleUnit(hidden_dim * 2, num_capsule=num_capsule, dim_capsule=dim_capsule, routings=routings)
        self.capsule_fc = Dense(num_capsule * dim_capsule, hidden_dim * 2, dropout=0.2, activation='relu')
        self.lstm_attention = Attention(hidden_dim * 2, seq_length)
        self.gru_attention = Attention(hidden_dim * 2, seq_length)

        fm_first_size = hidden_dim * 2 * 5
        fm_second_size = hidden_dim * 2 * sp.special.comb(5, 2)

        self.fc = nn.Linear(int(fm_first_size + fm_second_size), out_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(out_drop)
        self.output_layer = nn.Linear(out_hidden_dim, 1)

    def forward(self, inputs):
        x_mask = get_non_pad_mask(inputs)
        x_embed = self.embeddings(inputs)

        x_embed = self.embedding_dropout(torch.unsqueeze(x_embed, 0).transpose(1, 3))
        x_embed = torch.squeeze(x_embed.transpose(1, 3))

        x_lstm, _ = self.lstm(x_embed)
        x_gru, _ = self.gru(x_lstm)

        x_lstm = x_lstm * x_mask
        x_gru = x_gru * x_mask

        x_lstm_attention = self.lstm_attention(x_lstm)
        x_gru_attention = self.gru_attention(x_gru)
        x_avg_pool = torch.mean(x_gru, 1)
        x_max_pool, _ = torch.max(x_gru, 1)

        x_capsule = self.capsule(x_gru)
        x_capsule = self.capsule_fc(x_capsule)

        fm_first = [
            x_lstm_attention,
            x_gru_attention,
            x_avg_pool,
            x_max_pool,
            x_capsule
        ]

        fm_second = []
        for t_1, t_2 in itertools.combinations(fm_first, 2):
            fm_second.append(t_1 * t_2)

        x_fc = self.fc(torch.cat(fm_first + fm_second, 1))
        x_fc = self.dropout(self.relu(x_fc))
        outputs = self.output_layer(x_fc)
        return outputs
