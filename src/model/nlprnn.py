import itertools
import scipy as sp
import torch
import torch.nn as nn

from .common import Attention, Dense


class NLPFeaturesRNN(nn.Module):
    def __init__(self, input_shapes, embedding_matrix, seq_len, hidden_size=64, out_hidden_dim=64,
                 nlp_hidden_dim=64, nlp_dropout=0.2, embed_drop=0.2, out_drop=0.3, mask=False,
                 num_dense_layers=1, dense_activate='relu', embed_dropout_direction=0, factorize=False):
        super(NLPFeaturesRNN, self).__init__()

        self.mask = mask
        self.factorize = factorize

        continuous = []
        for i in range(num_dense_layers):
            if i == 0:
                layers = nn.ModuleList([
                    Dense(1, nlp_hidden_dim, dropout=nlp_dropout, activation=dense_activate)
                    for _ in range(input_shapes['continuous'])
                ])
            else:
                layers = nn.ModuleList([
                    Dense(nlp_hidden_dim, nlp_hidden_dim, dropout=nlp_dropout, activation=dense_activate)
                    for _ in range(input_shapes['continuous'])
                ])
            continuous.append(layers)

        self.embedding_continuous = nn.ModuleList(continuous)

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(embed_drop)
        self.embed_dropout_direction = embed_dropout_direction

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, seq_len)
        self.gru_attention = Attention(hidden_size * 2, seq_len)

        fm_first_size_nlp = nlp_hidden_dim * input_shapes['continuous']
        fm_second_size_nlp = nlp_hidden_dim * sp.special.comb(input_shapes['continuous'])

        fm_first_size_seq = hidden_size * 2 * 4
        fm_second_size_seq = hidden_size * 2 * 4

        if factorize:
            fc_size = fm_first_size_seq + fm_second_size_seq + fm_first_size_nlp + fm_second_size_nlp
        else:
            fc_size = fm_first_size_seq + fm_second_size_seq + fm_first_size_nlp

        self.fc = nn.Linear(int(fc_size), out_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(out_drop)
        self.output_layer = nn.Linear(out_hidden_dim, 1)

    def forward(self, inputs):
        x_embedding = self.embedding(inputs['text'])  # B x L x D
        if self.embed_drop_direction == 0:
            x_embedding = self.embedding_dropout(torch.unsqueeze(x_embedding, 0).transpose(1, 3))
            x_embedding = torch.squeeze(x_embedding.transpose(1, 3))
        else:
            x_embedding = self.embedding_dropout(x_embedding)

        x_lstm, _ = self.lstm(x_embedding)
        x_gru, _ = self.gru(x_lstm)

        if self.mask:
            x_lstm_attention = self.lstm_attention(x_lstm, inputs['mask'])
            x_gru_attention = self.gru_attention(x_gru, inputs['mask'])
            x_avg_pool = torch.mean(x_gru * inputs['mask'].unsqueeze(-1), 1)
            x_max_pool, _ = torch.max(x_gru * inputs['mask'].unsqueeze(-1), 1)
        else:
            x_lstm_attention = self.lstm_attention(x_lstm)
            x_gru_attention = self.gru_attention(x_gru)
            x_avg_pool = torch.mean(x_gru, 1)
            x_max_pool, _ = torch.max(x_gru, 1)

        x_continuous = []
        for i, layers in enumerate(self.embedding_continuous):
            if i == 0:
                continuous_inputs = inputs['continuous']
            else:
                continuous_inputs = x_continuous
                x_continuous = []
            for x, layer in zip(continuous_inputs, layers):
                x_continuous.append(layer(x))

        x_continuous_second = []
        if self.factorize:
            for t_1, t_2 in itertools.combinations(x_continuous, 2):
                x_continuous_second.append(t_1 * t_2)

        fm_first = [
            x_lstm_attention,
            x_gru_attention,
            x_avg_pool,
            x_max_pool
        ]

        fm_second = []
        for t_1, t_2 in itertools.combinations(fm_first, 2):
            fm_second.append(t_1 * t_2)

        x_fc = self.fc(torch.cat(fm_first + fm_second + x_continuous + x_continuous_second, 1))
        x_fc = self.dropout(self.relu(x_fc))
        outputs = self.output_layer(x_fc)
        return outputs


class NLPFeaturesRNNFM(nn.Module):
    def __init__(self, input_shapes, embedding_matrix, seq_len, hidden_size=64, out_hidden_dim=64,
                 embed_drop=0.2, out_drop=0.3, mask=False, num_dense_layers=1, dense_activate='relu'):
        super(NLPFeaturesRNNFM, self).__init__()

        self.mask = mask

        continuous = []
        for i in range(num_dense_layers):
            if i == 0:
                layers = nn.ModuleList([Dense(1, hidden_size * 2, dropout=0.3, activation=dense_activate)
                                        for _ in range(input_shapes['continuous'])])
            else:
                layers = nn.ModuleList([Dense(hidden_size * 2, hidden_size * 2, dropout=0.3, activation=dense_activate)
                                        for _ in range(input_shapes['continuous'])])
            continuous.append(layers)

        self.embedding_continuous = nn.ModuleList(continuous)

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(embed_drop)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, seq_len)
        self.gru_attention = Attention(hidden_size * 2, seq_len)

        fm_first_size = hidden_size * 2 * (4 + input_shapes['continuous'])
        fm_second_size = hidden_size * 2 * sp.special.comb(4 + input_shapes['continuous'], 2)

        self.fc = nn.Linear(int(fm_first_size + fm_second_size), out_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(out_drop)
        self.output_layer = nn.Linear(out_hidden_dim, 1)

    def forward(self, inputs):
        x_embedding = self.embedding(inputs['text'])  # B x L x D
        x_embedding = self.embedding_dropout(torch.unsqueeze(x_embedding, 0).transpose(1, 3))
        x_embedding = torch.squeeze(x_embedding.transpose(1, 3))

        x_lstm, _ = self.lstm(x_embedding)
        x_gru, _ = self.gru(x_lstm)

        if self.mask:
            x_lstm_attention = self.lstm_attention(x_lstm, inputs['mask'])
            x_gru_attention = self.gru_attention(x_gru, inputs['mask'])
            x_avg_pool = torch.mean(x_gru * inputs['mask'].unsqueeze(-1), 1)
            x_max_pool, _ = torch.max(x_gru * inputs['mask'].unsqueeze(-1), 1)
        else:
            x_lstm_attention = self.lstm_attention(x_lstm)
            x_gru_attention = self.gru_attention(x_gru)
            x_avg_pool = torch.mean(x_gru, 1)
            x_max_pool, _ = torch.max(x_gru, 1)

        x_continuous = []
        for i, layers in enumerate(self.embedding_continuous):
            if i == 0:
                continuous_inputs = inputs['continuous']
            else:
                continuous_inputs = x_continuous
                x_continuous = []
            for x, layer in zip(continuous_inputs, layers):
                x_continuous.append(layer(x))

        fm_first = [
            x_lstm_attention,
            x_gru_attention,
            x_avg_pool,
            x_max_pool
        ] + x_continuous

        fm_second = []
        for t_1, t_2 in itertools.combinations(fm_first, 2):
            fm_second.append(t_1 * t_2)

        x_fc = self.fc(torch.cat(fm_first + fm_second, 1))
        x_fc = self.dropout(self.relu(x_fc))
        outputs = self.output_layer(x_fc)
        return outputs
