import itertools
import scipy as sp
import torch
import torch.nn as nn

from .common import GeneralAttention, Attention
from .transformer import get_non_pad_mask


class StackedDeepRNN(nn.Module):
    def __init__(self, embedding_matrix, seq_len, embed_drop=0.1,
                 upper_layer_types=({'dim': 64, 'dropout': 0.3},),
                 rnn_layer_types=({'type': 'lstm', 'dim': 64, 'num_layers': 1, 'dropout': 0.0},
                                  {'type': 'gru', 'dim': 64, 'num_layers': 1, 'dropout': 0.0})):
        super(StackedDeepRNN, self).__init__()

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(embed_drop)

        rnn_layers = []
        for i, layer_type in enumerate(rnn_layer_types):
            if i == 0:
                input_dim = embedding_matrix.shape[1]
            else:
                input_dim = rnn_layer_types[i - 1]['dim'] * 2

            hidden_dim = layer_type['dim']
            num_layers = layer_type['num_layers']
            recurrent_drop = layer_type['dropout']

            if layer_type['type'] == 'lstm':
                rnn_layers.append(nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True,
                                          dropout=recurrent_drop, num_layers=num_layers))
            else:
                rnn_layers.append(nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True,
                                         dropout=recurrent_drop, num_layers=num_layers))

        self.rnn_layers = nn.ModuleList(rnn_layers)
        rnn_out_dim = rnn_layer_types[-1]['dim'] * 2

        attention_layers = []
        for _ in range(len(rnn_layers)):
            attention_layers.append(Attention(rnn_out_dim, seq_len))

        self.attention_layers = nn.ModuleList(attention_layers)

        num_fm_vectors = len(self.attention_layers) + 2
        fm_first_size = rnn_out_dim * 2 * num_fm_vectors
        fm_second_size = rnn_out_dim * 2 * sp.special.comb(num_fm_vectors, 2)

        upper_layers = []
        for i, layer_type in enumerate(upper_layer_types):
            if i == 0:
                input_dim = fm_first_size + fm_second_size
            else:
                input_dim = upper_layer_types[i - 1]['dim']
            upper_layers.append(nn.Linear(input_dim, layer_type['dim']))
            upper_layers.append(nn.ReLU())
            upper_layers.append(nn.Dropout(layer_type['dropout']))

        self.upper_layers = nn.ModuleList(upper_layers)

        upper_out_dim = upper_layer_types[-1]['dim']
        self.output_layer = nn.Linear(upper_out_dim, 1)

    def forward(self, inputs):
        x_embedding = self.embedding(inputs)  # B x L x D
        x_embedding = self.embedding_dropout(torch.unsqueeze(x_embedding, 0).transpose(1, 3))
        x_embedding = torch.squeeze(x_embedding.transpose(1, 3))

        x_mask = get_non_pad_mask(inputs)

        x_rnn = []
        x = x_embedding
        for i, rnn_layer in enumerate(self.rnn_layers):
            x, _ = rnn_layer(x)
            x_rnn.append(x)

        fm_first = []
        for x, attention_layer in zip(x_rnn, self.attention_layers):
            x_attention = attention_layer(x * x_mask)
            fm_first.append(x_attention)

        x_avg_pool = torch.mean(x_rnn[-1], 1)
        x_max_pool, _ = torch.max(x_rnn[-1], 1)

        fm_first.append(x_avg_pool)
        fm_first.append(x_max_pool)

        fm_second = []
        for t_1, t_2 in itertools.combinations(fm_first, 2):
            fm_second.append(t_1 * t_2)

        x_upper = torch.cat(fm_first + fm_second, 1)
        for i, upper_layer in enumerate(self.upper_layers):
            x_upper = upper_layer(x_upper)

        outputs = self.output_layer(x_upper)
        return outputs


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

        self.lstm_attention = GeneralAttention(hidden_size * 2, attention_type)
        self.gru_attention = GeneralAttention(hidden_size * 2, attention_type)

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

        x_lstm_attention, _ = self.lstm_attention(x_lstm, x_lstm)
        x_gru_attention, _ = self.gru_attention(x_gru, x_gru)

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


class AttentionMaskRNNAnother(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=64, out_hidden_dim=32, attention_type='general',
                 embed_drop=0.1, out_drop=0.2, mask=False):
        super(AttentionMaskRNNAnother, self).__init__()

        self.mask = mask

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(embed_drop)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = GeneralAttention(hidden_size * 2, attention_type)
        self.gru_attention = GeneralAttention(hidden_size * 2, attention_type)

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
        x_lstm, _ = self.lstm_attention(x_lstm, x_lstm)
        if self.mask:
            x_lstm = x_lstm * inputs['mask'].unsqueeze(-1)
        x_gru, _ = self.gru(x_lstm)
        x_gru, _ = self.gru_attention(x_gru, x_gru)
        if self.mask:
            x_gru = x_gru * inputs['mask'].unsqueeze(-1)

        x_avg_pool_lstm = torch.mean(x_lstm, 1)
        x_max_pool_lstm, _ = torch.max(x_lstm, 1)
        x_avg_pool_gru = torch.mean(x_gru, 1)
        x_max_pool_gru, _ = torch.max(x_gru, 1)

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