import itertools
import scipy as sp
import torch
import torch.nn as nn

from .common import Attention, Dense, GeneralAttention
from .transformer import get_sinusoid_encoding_table, get_non_pad_mask


class TransformerRNN(nn.Module):
    def __init__(self, embedding_matrix, seq_length, hidden_dim=64, out_hidden_dim=64, out_drop=0.3, embed_drop=0.2,
                 trans_drop=0.2, attention_type='dot', num_layers=2):
        super(TransformerRNN, self).__init__()

        n_position = seq_length + 1
        embed_dim = embedding_matrix.shape[1]

        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        self.position_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, embed_dim, padding_idx=0),
            freeze=True)
        self.embedding_dropout = nn.Dropout2d(embed_drop)

        self.transformer_layers = nn.ModuleList([
            GeneralAttention(embed_dim, attention_type=attention_type)
        ] for _ in range(num_layers))
        self.transformer_avg_fc = Dense(embed_dim, hidden_dim * 2, dropout=trans_drop, activation='relu')
        self.transformer_max_fc = Dense(embed_dim, hidden_dim * 2, dropout=trans_drop, activation='relu')

        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.lstm_attention = Attention(hidden_dim * 2, seq_length)
        self.gru_attention = Attention(hidden_dim * 2, seq_length)

        fm_first_size = hidden_dim * 2 * 8
        fm_second_size = hidden_dim * 2 * sp.special.comb(8, 2)

        self.fc = nn.Linear(int(fm_first_size + fm_second_size), out_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(out_drop)
        self.output_layer = nn.Linear(out_hidden_dim, 1)

    def forward(self, inputs):
        seq, pos = inputs['sequence'], inputs['position']
        non_pad_mask = get_non_pad_mask(seq)

        x_embed = self.word_embedding(seq)
        x_pos = self.position_embedding(pos)

        x_trans = x_embed + x_pos
        for layer in self.transformer_layers:
            x_trans, _ = layer(x_trans, x_trans)
            x_trans = x_trans *non_pad_mask

        x_embed_rnn = self.embedding_dropout(torch.unsqueeze(x_embed, 0).transpose(1, 3))
        x_embed_rnn = torch.squeeze(x_embed_rnn.transpose(1, 3))

        x_lstm, _ = self.lstm(x_embed_rnn)
        x_gru, _ = self.gru(x_lstm)

        x_trans_avg_pool = torch.mean(x_trans, 1)
        x_trans_max_pool, _ = torch.max(x_trans, 1)
        x_trans_avg_pool = self.transformer_avg_fc(x_trans_avg_pool)
        x_trans_max_pool = self.transformer_max_fc(x_trans_max_pool)

        x_lstm_attention = self.lstm_attention(x_lstm * non_pad_mask)
        x_gru_attention = self.gru_attention(x_gru * non_pad_mask)
        x_avg_pool_lstm = torch.mean(x_lstm, 1)
        x_max_pool_lstm, _ = torch.max(x_lstm, 1)
        x_avg_pool_gru = torch.mean(x_gru, 1)
        x_max_pool_gru, _ = torch.max(x_gru, 1)

        fm_first = [
            x_trans_avg_pool,
            x_trans_max_pool,
            x_lstm_attention,
            x_gru_attention,
            x_avg_pool_lstm,
            x_max_pool_lstm,
            x_avg_pool_gru,
            x_max_pool_gru
        ]

        fm_second = []
        for t_1, t_2 in itertools.combinations(fm_first, 2):
            fm_second.append(t_1 * t_2)

        x_fc = self.fc(torch.cat(fm_first + fm_second, 1))
        x_fc = self.dropout(self.relu(x_fc))
        outputs = self.output_layer(x_fc)
        return outputs
