import itertools
import scipy as sp
import torch
import torch.nn as nn

from .common import Attention


class StackedRNNFM(nn.Module):
    def __init__(self, embedding_matrix, seq_len, hidden_size=64, out_hidden_dim=32,
                 embed_drop=0.1, recurrent_drop=0.0, out_drop=0.2):
        super(StackedRNNFM, self).__init__()

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(embed_drop)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, dropout=recurrent_drop,
                            bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, dropout=recurrent_drop,
                          bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, seq_len)
        self.gru_attention = Attention(hidden_size * 2, seq_len)

        fm_first_size = hidden_size * 2 * 4
        fm_second_size = hidden_size * 2 * sp.special.comb(4, 2)

        self.fc = nn.Linear(int(fm_first_size + fm_second_size), out_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(out_drop)
        self.output_layer = nn.Linear(out_hidden_dim, 1)

    def forward(self, inputs):
        x_embedding = self.embedding(inputs)  # B x L x D
        x_embedding = self.embedding_dropout(torch.unsqueeze(x_embedding, 0).transpose(1, 3))
        x_embedding = torch.squeeze(x_embedding.transpose(1, 3))

        x_lstm, _ = self.lstm(x_embedding)
        x_gru, _ = self.gru(x_lstm)

        x_lstm_attention = self.lstm_attention(x_lstm)
        x_gru_attention = self.gru_attention(x_gru)
        x_avg_pool = torch.mean(x_gru, 1)
        x_max_pool, _ = torch.max(x_gru, 1)

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


class StackedDeepRNN(nn.Module):
    def __init__(self, embedding_matrix, seq_len, hidden_size=32, rnn_layer_types=('lstm', 'lstm'), out_hidden_dim=64,
                 embed_drop=0.1, seq_drop=0.2, layer_norm=False, factorize=True, out_drop=0.3, recurrent_drop=0.2):
        super(StackedDeepRNN, self).__init__()

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(embed_drop)

        rnn_layers = []
        for i, layer_type in enumerate(rnn_layer_types):
            if i == 0:
                h_dim = embedding_matrix.shape[1]
            else:
                h_dim = hidden_size * 2

            if layer_type == 'lstm':
                rnn_layers.append(nn.LSTM(h_dim, hidden_size, bidirectional=True, batch_first=True,
                                          dropout=recurrent_drop))
            else:
                rnn_layers.append(nn.GRU(h_dim, hidden_size, bidirectional=True, batch_first=True,
                                         dropout=recurrent_drop))

        rnn_norm_layers = []
        for _ in range(len(rnn_layers)):
            rnn_norm_layers.append(nn.LayerNorm(hidden_size * 2))
        self.layer_norm = layer_norm

        self.rnn_layers = nn.ModuleList(rnn_layers)
        self.rnn_norm_layers = nn.ModuleList(rnn_norm_layers)

        attention_layers = []
        for _ in range(len(rnn_layers)):
            attention_layers.append(Attention(hidden_size * 2, seq_len))

        self.attention_layers = nn.ModuleList(attention_layers)

        num_fm_vectors = len(self.attention_layers) + 2
        fm_first_size = hidden_size * 2 * num_fm_vectors
        self.factorize = factorize
        if factorize:
            fm_second_size = hidden_size * 2 * sp.special.comb(num_fm_vectors, 2)
        else:
            fm_second_size = 0

        self.fm_dropout_layers = nn.ModuleList([nn.Dropout(seq_drop) for _ in range(num_fm_vectors)])
        self.fc = nn.Linear(int(fm_first_size + fm_second_size), out_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(out_drop)
        self.output_layer = nn.Linear(out_hidden_dim, 1)

    def forward(self, inputs):
        x_embedding = self.embedding(inputs)  # B x L x D
        x_embedding = self.embedding_dropout(torch.unsqueeze(x_embedding, 0).transpose(1, 3))
        x_embedding = torch.squeeze(x_embedding.transpose(1, 3))

        x_rnn = []
        x = None
        for i, (rnn_layer, norm_layer) in enumerate(zip(self.rnn_layers, self.rnn_norm_layers)):
            if i == 0:
                x, _ = rnn_layer(x_embedding)
            else:
                x, _ = rnn_layer(x)
            if self.layer_norm:
                x = norm_layer(x)
            x_rnn.append(x)

        fm_first = []
        for x, attention_layer in zip(x_rnn, self.attention_layers):
            x_attention = attention_layer(x)
            fm_first.append(x_attention)

        x_avg_pool = torch.mean(x_rnn[-1], 1)
        x_max_pool, _ = torch.max(x_rnn[-1], 1)

        fm_first.append(x_avg_pool)
        fm_first.append(x_max_pool)

        fm_first = [drop(x) for x, drop in zip(fm_first, self.fm_dropout_layers)]

        fm_second = []
        if self.factorize:
            for t_1, t_2 in itertools.combinations(fm_first, 2):
                fm_second.append(t_1 * t_2)

        x_fc = self.fc(torch.cat(fm_first + fm_second, 1))
        x_fc = self.dropout(self.relu(x_fc))
        outputs = self.output_layer(x_fc)
        return outputs


class StackedRNNFastFM(nn.Module):
    def __init__(self, embedding_matrix, seq_len, hidden_size=64, out_hidden_dim=32,
                 embed_drop=0.1, recurrent_drop=0.0, out_drop=0.2):
        super(StackedRNNFastFM, self).__init__()

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(embed_drop)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, dropout=recurrent_drop,
                            bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, dropout=recurrent_drop,
                          bidirectional=True, batch_first=True)

        self.fast_attention = Attention(embedding_matrix.shape[1], seq_len)
        self.fast_dropout = nn.Dropout(0.2)
        self.fast_fc = nn.Linear(embedding_matrix.shape[1], hidden_size * 2)
        self.fast_relu = nn.ReLU()
        self.lstm_attention = Attention(hidden_size * 2, seq_len)
        self.gru_attention = Attention(hidden_size * 2, seq_len)

        fm_first_size = hidden_size * 2 * 5
        fm_second_size = hidden_size * 2 * sp.special.comb(5, 2)

        self.fc = nn.Linear(int(fm_first_size + fm_second_size), out_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(out_drop)
        self.output_layer = nn.Linear(out_hidden_dim, 1)

    def forward(self, inputs):
        x_embedding = self.embedding(inputs)  # B x L x D
        x_embedding = self.embedding_dropout(torch.unsqueeze(x_embedding, 0).transpose(1, 3))
        x_embedding = torch.squeeze(x_embedding.transpose(1, 3))

        x_lstm, _ = self.lstm(x_embedding)
        x_gru, _ = self.gru(x_lstm)

        x_fast_attention = self.fast_attention(x_embedding)
        x_fast_attention = self.fast_dropout(self.fast_relu(self.fast_fc(x_fast_attention)))
        x_lstm_attention = self.lstm_attention(x_lstm)
        x_gru_attention = self.gru_attention(x_gru)
        x_avg_pool = torch.mean(x_gru, 1)
        x_max_pool, _ = torch.max(x_gru, 1)

        fm_first = [
            x_fast_attention,
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


class StackedNormalizedRNNFM(nn.Module):
    def __init__(self, embedding_matrix, seq_len, hidden_size=64, out_hidden_dim=32,
                 embed_drop=0.1, residual=False, out_drop=0.2):
        super(StackedNormalizedRNNFM, self).__init__()

        self.residual = residual

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(embed_drop)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True, batch_first=True)
        self.lstm_norm = nn.LayerNorm(hidden_size * 2)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.gru_norm = nn.LayerNorm(hidden_size * 2)

        self.lstm_attention = Attention(hidden_size * 2, seq_len)
        self.gru_attention = Attention(hidden_size * 2, seq_len)

        fm_first_size = hidden_size * 2 * 4
        fm_second_size = hidden_size * 2 * sp.special.comb(4, 2)

        self.fc = nn.Linear(int(fm_first_size + fm_second_size), out_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(out_drop)
        self.output_layer = nn.Linear(out_hidden_dim, 1)

    def forward(self, inputs):
        x_embedding = self.embedding(inputs)  # B x L x D
        x_embedding = self.embedding_dropout(torch.unsqueeze(x_embedding, 0).transpose(1, 3))
        x_embedding = torch.squeeze(x_embedding.transpose(1, 3))

        x_lstm, _ = self.lstm(x_embedding)
        x_lstm = self.lstm_norm(x_lstm)
        x_gru, _ = self.gru(x_lstm)
        if self.residual:
            x_gru += x_lstm
        x_gru = self.gru_norm(x_gru)

        x_lstm_attention = self.lstm_attention(x_lstm)
        x_gru_attention = self.gru_attention(x_gru)
        x_avg_pool = torch.mean(x_gru, 1)
        x_max_pool, _ = torch.max(x_gru, 1)

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


class StackedMaskRNNFM(nn.Module):
    def __init__(self, embedding_matrix, seq_len, hidden_size=64, out_hidden_dim=32,
                 embed_drop=0.1, recurrent_drop=0.0, out_drop=0.2):
        super(StackedMaskRNNFM, self).__init__()

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(embed_drop)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, dropout=recurrent_drop,
                            bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, dropout=recurrent_drop,
                          bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, seq_len)
        self.gru_attention = Attention(hidden_size * 2, seq_len)

        fm_first_size = hidden_size * 2 * 4
        fm_second_size = hidden_size * 2 * sp.special.comb(4, 2)

        self.fc = nn.Linear(int(fm_first_size + fm_second_size), out_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(out_drop)
        self.output_layer = nn.Linear(out_hidden_dim, 1)

    def forward(self, inputs):
        x_embedding = self.embedding(inputs['sequence'])  # B x L x D
        x_embedding = self.embedding_dropout(torch.unsqueeze(x_embedding, 0).transpose(1, 3))
        x_embedding = torch.squeeze(x_embedding.transpose(1, 3))

        x_lstm, _ = self.lstm(x_embedding)
        x_gru, _ = self.gru(x_lstm)

        x_lstm_attention = self.lstm_attention(x_lstm, inputs['mask'])
        x_gru_attention = self.gru_attention(x_gru, inputs['mask'])
        x_avg_pool = torch.mean(x_gru * inputs['mask'].unsqueese(-1), 1)
        x_max_pool, _ = torch.max(x_gru * inputs['mask'].unsqueese(-1), 1)

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
