import itertools
import scipy as sp
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .common import Dense, Attention
from .transformer import get_non_pad_mask


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalConvolutionBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalConvolutionBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvolutionUnit(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvolutionUnit, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalConvolutionBlock(in_channels, out_channels, kernel_size,
                                                stride=1, dilation=dilation_size,
                                                padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NLPFeaturesTCN(nn.Module):
    def __init__(self, embedding_matrix, seq_len, nlp_size, embed_drop=0.2, mask=True,
                 nlp_layer_types=({'activation': 'relu', 'dim': 16, 'dropout': 0.2},
                                  {'activation': 'relu', 'dim': 16, 'dropout': 0.2}),
                 tcn_layer_types=({'num_channels': [256, 256, 300], 'kernel_size': 2, 'dropout': 0.2},
                                  {'num_channels': [256, 256, 300], 'kernel_size': 3, 'dropout': 0.2},
                                  {'num_channels': [256, 256, 300], 'kernel_size': 4, 'dropout': 0.2}),
                 upper_layer_types=({'dim': 64, 'dropout': 0.3},)):
        super(NLPFeaturesTCN, self).__init__()

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

        tcn_layers = []
        for i, layer_type in enumerate(tcn_layer_types):
            tcn_layers.append(TemporalConvolutionUnit(embedding_matrix.shape[1],
                                                      num_channels=layer_type['num_channels'],
                                                      kernel_size=layer_type['kernel_size'],
                                                      dropout=layer_type['dropout']))

        self.tcn_layers = nn.ModuleList(tcn_layers)
        tcn_out_dim = tcn_layer_types[0]['num_channels'][-1]

        attention_layers = []
        for _ in tcn_layer_types:
            attention_layers.append(Attention(tcn_out_dim, seq_len))

        self.attention_layers = nn.ModuleList(attention_layers)

        first_order_rnn_dim = tcn_out_dim * (3 * len(tcn_layers))
        second_order_rnn_dim = tcn_out_dim * sp.special.comb(3 * len(tcn_layers), 2)

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

        x_tcn = []
        for i, tcn_layer in enumerate(self.tcn_layers):
            x = tcn_layer(x_embedding.transpose(1, 2)).transpose(1, 2)
            x_tcn.append(x)

        x_tcn_attention = []
        for x, attention_layer in zip(x_tcn, self.attention_layers):
            x_attention = attention_layer(x, x_mask.squeeze(-1))
            x_tcn_attention.append(x_attention)

        x_tcn_avg_pool = []
        for x in x_tcn:
            x_tcn_avg_pool.append(torch.mean(x * x_mask, 1))

        x_tcn_max_pool = []
        for x in x_tcn:
            x_tcn_max_pool.append(torch.max(x * x_mask, 1)[0])

        x_first_order = x_tcn_attention + x_tcn_avg_pool + x_tcn_max_pool

        x_second_order = []
        for t_1, t_2 in itertools.combinations(x_first_order, 2):
            x_second_order.append(t_1 * t_2)

        x_upper = torch.cat(x_first_order + x_second_order + x_nlp, 1)
        for i, upper_layer in enumerate(self.upper_layers):
            x_upper = upper_layer(x_upper)

        outputs = self.output_layer(x_upper)
        return outputs
