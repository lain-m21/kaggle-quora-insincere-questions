import itertools
import scipy as sp
import torch
import torch.nn as nn

from .common import Dense, Attention
from .transformer import get_non_pad_mask
from .tcn import TemporalConvolutionUnit


class NLPFeaturesTCNRNN(nn.Module):
    def __init__(self, embedding_matrix, seq_len, nlp_size, embed_drop=0.2, mask=True,
                 nlp_layer_types=({'activation': 'relu', 'dim': 16, 'dropout': 0.2},
                                  {'activation': 'relu', 'dim': 16, 'dropout': 0.2}),
                 tcn_layer_types=({'num_channels': [16, 16, 16], 'kernel_size': 2, 'dropout': 0.2},
                                  {'num_channels': [16, 16, 16], 'kernel_size': 3, 'dropout': 0.2},
                                  {'num_channels': [16, 16, 16], 'kernel_size': 4, 'dropout': 0.2}),
                 rnn_layer_types=({'type': 'lstm', 'dim': 64, 'num_layers': 1, 'dropout': 0.0},
                                  {'type': 'gru', 'dim': 64, 'num_layers': 1, 'dropout': 0.0}),
                 upper_layer_types=({'dim': 64, 'dropout': 0.3},)):
        super(NLPFeaturesTCNRNN, self).__init__()

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

        tcn_layers = []
        for i, layer_type in enumerate(tcn_layer_types):
            tcn_layers.append(TemporalConvolutionUnit(embedding_matrix.shape[1],
                                                      num_channels=layer_type['num_channels'],
                                                      kernel_size=layer_type['kernel_size'],
                                                      dropout=layer_type['dropout']))

        self.tcn_layers = nn.ModuleList(tcn_layers)
        tcn_out_dim = tcn_layer_types[0]['num_channels'][-1]

        rnn_attention_layers = []
        for _ in rnn_layer_types:
            rnn_attention_layers.append(Attention(rnn_out_dim, seq_len))

        self.rnn_attention_layers = nn.ModuleList(rnn_attention_layers)

        tcn_attention_layers = []
        for _ in tcn_layer_types:
            tcn_attention_layers.append(Attention(tcn_out_dim, seq_len))

        self.tcn_attention_layers = nn.ModuleList(tcn_attention_layers)

        first_order_dim = rnn_out_dim * (2 + len(rnn_layer_types)) + tcn_out_dim * (3 * len(tcn_layers))
        second_order_dim = rnn_out_dim * sp.special.comb((2 + len(rnn_layer_types)), 2)

        upper_layers = []
        for i, layer_type in enumerate(upper_layer_types):
            if i == 0:
                input_dim = int(first_order_dim + second_order_dim + nlp_out_dim)
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
        for x, attention_layer in zip(x_rnn, self.rnn_attention_layers):
            x_attention = attention_layer(x, x_mask.squeeze(-1))
            x_rnn_attention.append(x_attention)

        x_rnn_avg_pool = torch.mean(x_rnn[-1] * x_mask, 1)
        x_rnn_max_pool, _ = torch.max(x_rnn[-1] * x_mask, 1)

        x_tcn = []
        for i, tcn_layer in enumerate(self.tcn_layers):
            x = tcn_layer(x_embedding.transpose(1, 2)).transpose(1, 2)
            x_tcn.append(x)

        x_tcn_attention = []
        for x, attention_layer in zip(x_tcn, self.tcn_attention_layers):
            x_attention = attention_layer(x, x_mask.squeeze(-1))
            x_tcn_attention.append(x_attention)

        x_tcn_avg_pool = []
        for x in x_tcn:
            x_tcn_avg_pool.append(torch.mean(x * x_mask, 1))

        x_tcn_max_pool = []
        for x in x_tcn:
            x_tcn_max_pool.append(torch.max(x * x_mask, 1)[0])

        x_first_order = [
            x_rnn_avg_pool,
            x_rnn_max_pool
        ] + x_rnn_attention

        x_second_order = []
        for t_1, t_2 in itertools.combinations(x_first_order, 2):
            x_second_order.append(t_1 * t_2)

        x_first_order += x_tcn_attention + x_tcn_avg_pool + x_tcn_max_pool

        x_upper = torch.cat(x_first_order + x_second_order + x_nlp, 1)
        for i, upper_layer in enumerate(self.upper_layers):
            x_upper = upper_layer(x_upper)

        outputs = self.output_layer(x_upper)
        return outputs


class NLPFeaturesConcatTCNRNN(nn.Module):
    def __init__(self, embedding_matrix, seq_len, embed_drop=0.2, mask=True,
                 nlp_layer_types=({'activation': 'relu', 'dim': 128, 'dropout': 0.2},),
                 tcn_layer_types=({'num_channels': [16, 16, 16], 'kernel_size': 2, 'dropout': 0.2},
                                  {'num_channels': [16, 16, 16], 'kernel_size': 3, 'dropout': 0.2},
                                  {'num_channels': [16, 16, 16], 'kernel_size': 4, 'dropout': 0.2}),
                 rnn_layer_types=({'type': 'lstm', 'dim': 64, 'num_layers': 1, 'dropout': 0.0},
                                  {'type': 'gru', 'dim': 64, 'num_layers': 1, 'dropout': 0.0}),
                 upper_layer_types=({'dim': 64, 'dropout': 0.5},
                                    {'dim': 64, 'dropout': 0.3})):
        super(NLPFeaturesConcatTCNRNN, self).__init__()

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
            nlp_layers.append(Dense(input_dim, hidden_dim, dropout=dropout, activation=activation))

        self.nlp_layers = nn.ModuleList(nlp_layers)
        nlp_out_dim = nlp_layer_types[-1]['dim']

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

        tcn_layers = []
        for i, layer_type in enumerate(tcn_layer_types):
            tcn_layers.append(TemporalConvolutionUnit(embedding_matrix.shape[1],
                                                      num_channels=layer_type['num_channels'],
                                                      kernel_size=layer_type['kernel_size'],
                                                      dropout=layer_type['dropout']))

        self.tcn_layers = nn.ModuleList(tcn_layers)
        tcn_out_dim = tcn_layer_types[0]['num_channels'][-1]

        rnn_attention_layers = []
        for _ in rnn_layer_types:
            rnn_attention_layers.append(Attention(rnn_out_dim, seq_len))

        self.rnn_attention_layers = nn.ModuleList(rnn_attention_layers)

        tcn_attention_layers = []
        for _ in tcn_layer_types:
            tcn_attention_layers.append(Attention(tcn_out_dim, seq_len))

        self.tcn_attention_layers = nn.ModuleList(tcn_attention_layers)

        first_order_dim = rnn_out_dim * (2 + len(rnn_layer_types)) + tcn_out_dim * (3 * len(tcn_layers))
        second_order_dim = rnn_out_dim * sp.special.comb((2 + len(rnn_layer_types)), 2)

        upper_layers = []
        for i, layer_type in enumerate(upper_layer_types):
            if i == 0:
                input_dim = int(first_order_dim + second_order_dim + nlp_out_dim)
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
        for x, attention_layer in zip(x_rnn, self.rnn_attention_layers):
            x_attention = attention_layer(x, x_mask.squeeze(-1))
            x_rnn_attention.append(x_attention)

        x_rnn_avg_pool = torch.mean(x_rnn[-1] * x_mask, 1)
        x_rnn_max_pool, _ = torch.max(x_rnn[-1] * x_mask, 1)

        x_tcn = []
        for i, tcn_layer in enumerate(self.tcn_layers):
            x = tcn_layer(x_embedding.transpose(1, 2)).transpose(1, 2)
            x_tcn.append(x)

        x_tcn_attention = []
        for x, attention_layer in zip(x_tcn, self.tcn_attention_layers):
            x_attention = attention_layer(x, x_mask.squeeze(-1))
            x_tcn_attention.append(x_attention)

        x_tcn_avg_pool = []
        for x in x_tcn:
            x_tcn_avg_pool.append(torch.mean(x * x_mask, 1))

        x_tcn_max_pool = []
        for x in x_tcn:
            x_tcn_max_pool.append(torch.max(x * x_mask, 1)[0])

        x_first_order = [
            x_rnn_avg_pool,
            x_rnn_max_pool
        ] + x_rnn_attention

        x_second_order = []
        for t_1, t_2 in itertools.combinations(x_first_order, 2):
            x_second_order.append(t_1 * t_2)

        x_first_order += x_tcn_attention + x_tcn_avg_pool + x_tcn_max_pool

        x_upper = torch.cat(x_first_order + x_second_order + x_nlp, 1)
        for i, upper_layer in enumerate(self.upper_layers):
            x_upper = upper_layer(x_upper)

        outputs = self.output_layer(x_upper)
        return outputs
