import itertools
import scipy as sp
import torch
import torch.nn as nn

from .common import Attention, Dense
from .transformer import get_non_pad_mask


class NLPFeaturesDeepRNN(nn.Module):
    def __init__(self, embedding_matrix, seq_len, nlp_size, embed_drop=0.2, mask=False,
                 nlp_layer_types=({'activation': 'relu', 'dim': 16, 'dropout': 0.2},
                                  {'activation': 'relu', 'dim': 16, 'dropout': 0.2}),
                 rnn_layer_types=({'type': 'lstm', 'dim': 64, 'num_layers': 1, 'dropout': 0.0},
                                  {'type': 'gru', 'dim': 64, 'num_layers': 1, 'dropout': 0.0}),
                 upper_layer_types=({'dim': 64, 'dropout': 0.3},)):
        super(NLPFeaturesDeepRNN, self).__init__()

        self.mask = mask
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(embed_drop)

        nlp_layers = []
        for i, layer_type in enumerate(nlp_layer_types):
            if i == 0:
                input_dim = 1
            else:
                input_dim = nlp_layer_types[i - 1]['dim']
            hidden_dim = layer_type['dim']
            dropout = layer_type['dropout']
            activation = layer_type['activation']
            nlp_layers.append(nn.ModuleList([
                Dense(input_dim, hidden_dim, dropout=dropout, activation=activation)
                for _ in range(nlp_size)
            ]))

        self.nlp_layers = nn.ModuleList(nlp_layers)
        nlp_out_dim = nlp_layer_types[-1]['dim'] * nlp_size

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
        for layer_type in rnn_layer_types:
            dim = layer_type['dim'] * 2
            attention_layers.append(Attention(dim, seq_len))

        self.attention_layers = nn.ModuleList(attention_layers)

        first_order_rnn_dim = sum([layer_type['dim'] * 2 for layer_type in rnn_layer_types])
        second_order_rnn_dim = rnn_out_dim * sp.special.comb(3, 2)

        import IPython
        IPython.embed()

        upper_layers = []
        for i, layer_type in enumerate(upper_layer_types):
            if i == 0:
                input_dim = int(first_order_rnn_dim + second_order_rnn_dim + nlp_out_dim)
            else:
                input_dim = upper_layer_types[i - 1]['dim']
            hidden_dim = layer_type['dim']
            dropout = layer_type['dropout']
            upper_layers.append(Dense(input_dim, hidden_dim, dropout, activation='relu'))

        self.upper_layers = nn.ModuleList(upper_layers)

        upper_out_dim = upper_layer_types[-1]['dim']
        self.output_layer = nn.Linear(upper_out_dim, 1)

    def forward(self, inputs):
        x_embedding = self.embedding(inputs['text'])  # B x L x D
        x_embedding = self.embedding_dropout(torch.unsqueeze(x_embedding, 0).transpose(1, 3))
        x_embedding = torch.squeeze(x_embedding.transpose(1, 3))

        x_mask = get_non_pad_mask(inputs['text'])

        x_nlp = inputs['nlp']
        for i, layers in enumerate(self.nlp_layers):
            x_nlp = [layer(x) for x, layer in zip(x_nlp, layers)]

        x_rnn = []
        x = x_embedding
        for i, rnn_layer in enumerate(self.rnn_layers):
            x, _ = rnn_layer(x)
            x_rnn.append(x)

        x_rnn_attention = []
        for x, attention_layer in zip(x_rnn, self.attention_layers):
            x_attention = attention_layer(x * x_mask)
            x_rnn_attention.append(x_attention)

        x_avg_pool = torch.mean(x_rnn[-1], 1)
        x_max_pool, _ = torch.max(x_rnn[-1], 1)

        x_first_order_rnn = [
            x_rnn_attention[-1],
            x_avg_pool,
            x_max_pool
        ]

        x_second_order_rnn = []
        for t_1, t_2 in itertools.combinations(x_first_order_rnn, 2):
            x_second_order_rnn.append(t_1 * t_2)

        x_first_order_rnn += x_rnn_attention[:-1]

        x_upper = torch.cat(x_first_order_rnn + x_second_order_rnn + x_nlp, 1)
        for i, upper_layer in enumerate(self.upper_layers):
            x_upper = upper_layer(x_upper)

        outputs = self.output_layer(x_upper)
        return outputs


class NLPFeaturesRNN(nn.Module):
    def __init__(self, input_shapes, embedding_matrix, seq_len, hidden_size=64, out_hidden_dim=64,
                 nlp_hidden_dim=64, nlp_dropout=0.2, embed_drop=0.2, out_drop=0.3, mask=False,
                 num_dense_layers=1, dense_activate='relu', embed_drop_direction=0, factorize=False):
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
        self.embed_drop_direction = embed_drop_direction

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, seq_len)
        self.gru_attention = Attention(hidden_size * 2, seq_len)

        fm_first_size_nlp = nlp_hidden_dim * input_shapes['continuous']
        fm_second_size_nlp = nlp_hidden_dim * sp.special.comb(input_shapes['continuous'], 2)

        fm_first_size_seq = hidden_size * 2 * 4
        fm_second_size_seq = hidden_size * 2 * sp.special.comb(4, 2)

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
            mask = get_non_pad_mask(inputs['text'])
            x_lstm_attention = self.lstm_attention(x_lstm, mask.squeeze(-1))
            x_gru_attention = self.gru_attention(x_gru, mask.squeeze(-1))
            x_avg_pool = torch.mean(x_gru * mask, 1)
            x_max_pool, _ = torch.max(x_gru * mask, 1)
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
