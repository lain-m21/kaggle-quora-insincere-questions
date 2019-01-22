import itertools
import scipy as sp
import torch
import torch.nn as nn

from .common import Attention, GeneralAttention


class BranchedMaskCNNRNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=64, out_hidden_dim=64, attention_type='general',
                 kernel_sizes=(2, 7), embed_drop=0.2, out_drop=0.3, pool_type='avg', mask=False):
        super(BranchedMaskCNNRNN, self).__init__()

        self.pool_type = pool_type
        self.mask = mask

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(embed_drop)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.cnn_layers = nn.ModuleList([nn.ModuleList([
            nn.Conv1d(hidden_size * 2, hidden_size * 2, kernel_size=k),
            nn.ReLU()
        ]) for k in range(*kernel_sizes)])

        self.lstm_attention = GeneralAttention(hidden_size * 2, attention_type)
        self.gru_attention = GeneralAttention(hidden_size * 2, attention_type)
        self.cnn_attention_layers = nn.ModuleList([GeneralAttention(hidden_size * 2, attention_type)
                                                   for _ in range(*kernel_sizes)])

        if pool_type == 'both':
            pool_factor = 2
        else:
            pool_factor = 1
        fm_input_size = 4 + pool_factor * len(list(range(*kernel_sizes)))
        fm_first_size = hidden_size * 2 * fm_input_size
        fm_second_size = hidden_size * 2 * sp.special.comb(fm_input_size, 2)

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
        x_cnn = []
        for layers in self.cnn_layers:
            x = layers[0](x_lstm.transpose(1, 2))
            x = layers[1](x)
            x_cnn.append(x.transpose(1, 2))

        x_lstm_attention, _ = self.lstm_attention(x_lstm, x_lstm)
        x_gru_attention, _ = self.gru_attention(x_gru, x_gru)
        x_cnn_attention  = []
        for x, layer in zip(x_cnn, self.cnn_attention_layers):
            x_attention, _ = layer(x, x)
            x_cnn_attention.append(x_attention)

        if self.mask:
            x_lstm_attention = x_lstm_attention * inputs['mask'].unsqueeze(-1)
            x_gru_attention = x_gru_attention * inputs['mask'].unsqueeze(-1)

        x_avg_pool_lstm = torch.mean(x_lstm_attention, 1)
        x_max_pool_lstm, _ = torch.max(x_lstm_attention, 1)
        x_avg_pool_gru = torch.mean(x_gru_attention, 1)
        x_max_pool_gru, _ = torch.max(x_gru_attention, 1)

        x_pool_cnn = []
        for x in x_cnn_attention:
            if self.pool_type == 'avg':
                x_pool = torch.mean(x, 1)
                x_pool_cnn.append(x_pool)
            elif self.pool_type == 'max':
                x_pool, _ = torch.max(x, 1)
                x_pool_cnn.append(x_pool)
            else:
                x_avg_pool = torch.mean(x, 1)
                x_max_pool, _ = torch.max(x, 1)
                x_pool_cnn.append(x_avg_pool)
                x_pool_cnn.append(x_max_pool)

        fm_first = [
            x_avg_pool_lstm,
            x_max_pool_lstm,
            x_avg_pool_gru,
            x_max_pool_gru
        ] + x_pool_cnn

        fm_second = []
        for t_1, t_2 in itertools.combinations(fm_first, 2):
            fm_second.append(t_1 * t_2)

        x_fc = self.fc(torch.cat(fm_first + fm_second, 1))
        x_fc = self.dropout(self.relu(x_fc))
        outputs = self.output_layer(x_fc)
        return outputs


class BranchedMaskCNNRNNAnother(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=64, out_hidden_dim=64, attention_type='general',
                 kernel_sizes=(2, 7), embed_drop=0.2, out_drop=0.3, pool_type='avg', mask=False):
        super(BranchedMaskCNNRNNAnother, self).__init__()

        self.pool_type = pool_type
        self.mask = mask

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(embed_drop)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.cnn_layers = nn.ModuleList([nn.ModuleList([
            nn.Conv1d(hidden_size * 2, hidden_size * 2, kernel_size=k),
            nn.ReLU()
        ]) for k in range(*kernel_sizes)])

        self.lstm_attention = GeneralAttention(hidden_size * 2, attention_type)
        self.gru_attention = GeneralAttention(hidden_size * 2, attention_type)
        self.cnn_attention_layers = nn.ModuleList([GeneralAttention(hidden_size * 2, attention_type)])

        if pool_type == 'both':
            pool_factor = 2
        else:
            pool_factor = 1
        fm_input_size = 4 + pool_factor * len(list(range(*kernel_sizes)))
        fm_first_size = hidden_size * 2 * fm_input_size
        fm_second_size = hidden_size * 2 * sp.special.comb(fm_input_size, 2)

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
        x_cnn = []
        for layers in self.cnn_layers:
            x = layers[0](x_lstm.transpose(1, 2))
            x = layers[1](x)
            x_cnn.append(x.transpose(1, 2))

        x_cnn_attention  = []
        for x, layer in zip(x_cnn, self.cnn_attention_layers):
            x_attention, _ = layer(x, x)
            x_cnn_attention.append(x_attention)

        x_avg_pool_lstm = torch.mean(x_lstm, 1)
        x_max_pool_lstm, _ = torch.max(x_lstm, 1)
        x_avg_pool_gru = torch.mean(x_gru, 1)
        x_max_pool_gru, _ = torch.max(x_gru, 1)

        x_pool_cnn = []
        for x in x_cnn_attention:
            if self.pool_type == 'avg':
                x_pool = torch.mean(x, 1)
                x_pool_cnn.append(x_pool)
            elif self.pool_type == 'max':
                x_pool, _ = torch.max(x, 1)
                x_pool_cnn.append(x_pool)
            else:
                x_avg_pool = torch.mean(x, 1)
                x_max_pool, _ = torch.max(x, 1)
                x_pool_cnn.append(x_avg_pool)
                x_pool_cnn.append(x_max_pool)

        fm_first = [
            x_avg_pool_lstm,
            x_max_pool_lstm,
            x_avg_pool_gru,
            x_max_pool_gru
        ] + x_pool_cnn

        fm_second = []
        for t_1, t_2 in itertools.combinations(fm_first, 2):
            fm_second.append(t_1 * t_2)

        x_fc = self.fc(torch.cat(fm_first + fm_second, 1))
        x_fc = self.dropout(self.relu(x_fc))
        outputs = self.output_layer(x_fc)
        return outputs


class StackedCNNRNN(nn.Module):
    def __init__(self, embedding_matrix, seq_len, hidden_size=32, kernel_sizes=(3, 5),
                 out_hidden_dim=32, seq_dropout=0.2, embed_drop=0.1, out_drop=0.2):
        super(StackedCNNRNN, self).__init__()

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(embed_drop)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True, batch_first=True)
        self.lstm_norm = nn.LayerNorm(hidden_size * 2)

        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.gru_norm = nn.LayerNorm(hidden_size * 2)

        self.cnn_layers = nn.ModuleList([nn.ModuleList([
            nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=k, padding=k // 2),
            nn.ReLU()
        ]) for k in kernel_sizes])

        self.lstm_attention = Attention(hidden_size * 2, seq_len)
        self.gru_attention = Attention(hidden_size * 2, seq_len)
        self.cnn_attention = Attention(hidden_size * len(kernel_sizes), seq_len)

        fm_first_size = hidden_size * 2 * 5
        fm_second_size = hidden_size * 2 * sp.special.comb(5, 2)

        self.fm_dropout_layers = nn.ModuleList([nn.Dropout(seq_dropout) for _ in range(5)])
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
        x_gru = self.gru_norm(x_gru)

        x_cnn = []
        for layers in self.cnn_layers:
            x = layers[0](x_gru.transpose(1, 2))
            x = layers[1](x)
            x_cnn.append(x)

        x_cnn = torch.cat(x_cnn, dim=1).transpose(1, 2)

        x_lstm_attention = self.lstm_attention(x_lstm)
        x_gru_attention = self.gru_attention(x_gru)
        x_cnn_attention = self.cnn_attention(x_cnn)
        x_avg_pool = torch.mean(x_cnn, 1)
        x_max_pool, _ = torch.max(x_cnn, 1)

        fm_first = [
            x_lstm_attention,
            x_gru_attention,
            x_cnn_attention,
            x_avg_pool,
            x_max_pool
        ]

        fm_first = [drop(x) for x, drop in zip(fm_first, self.fm_dropout_layers)]

        fm_second = []
        for t_1, t_2 in itertools.combinations(fm_first, 2):
            fm_second.append(t_1 * t_2)

        x_fc = self.fc(torch.cat(fm_first + fm_second, 1))
        x_fc = self.dropout(self.relu(x_fc))
        outputs = self.output_layer(x_fc)
        return outputs


class StackedBranchedCNNRNN(nn.Module):
    def __init__(self, embedding_matrix, seq_len, hidden_size=32, out_hidden_dim=32, seq_dropout=0.2,
                 embed_drop=0.1, out_drop=0.2):
        super(StackedBranchedCNNRNN, self).__init__()

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(embed_drop)

        self.cnn_layers_odd = nn.ModuleList([nn.ModuleList([
            nn.Conv1d(embedding_matrix.shape[1], hidden_size, kernel_size=k, padding=k//2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        ]) for k in [3, 5]])
        self.cnn_layers_even = nn.ModuleList([nn.ModuleList([
            nn.Conv1d(embedding_matrix.shape[1], hidden_size, kernel_size=k, padding=k//2-1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        ]) for k in [2, 4]])

        self.gru_odd = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.gru_even = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.gru_norm_odd = nn.LayerNorm(hidden_size * 2)
        self.gru_norm_even = nn.LayerNorm(hidden_size * 2)

        self.cnn_attention_odd = Attention(hidden_size * 2, seq_len)
        self.cnn_attention_even = Attention(hidden_size * 2, seq_len - 1)
        self.gru_attention_odd = Attention(hidden_size * 2, seq_len)
        self.gru_attention_even = Attention(hidden_size * 2, seq_len - 1)

        fm_first_size = hidden_size * 2 * 8
        fm_second_size = hidden_size * 2 * sp.special.comb(8, 2)
        self.fm_dropout_layers = nn.ModuleList([nn.Dropout(seq_dropout) for _ in range(8)])

        self.fc = nn.Linear(int(fm_first_size + fm_second_size), out_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(out_drop)
        self.output_layer = nn.Linear(out_hidden_dim, 1)

    def forward(self, inputs):
        x_embedding = self.embedding(inputs)  # B x L x D
        x_embedding = self.embedding_dropout(torch.unsqueeze(x_embedding, 0).transpose(1, 3))
        x_embedding = torch.squeeze(x_embedding.transpose(1, 3))

        x_embedding = x_embedding.transpose(1, 2)  # B x L x D -> B x D x L

        x_cnn_odd = []
        for layers in self.cnn_layers_odd:
            x = layers[0](x_embedding)
            x = layers[1](x)
            x = layers[2](x)
            x_cnn_odd.append(x)

        x_cnn_even = []
        for layers in self.cnn_layers_even:
            x = layers[0](x_embedding)
            x = layers[1](x)
            x = layers[2](x)
            x_cnn_even.append(x)

        x_cnn_odd = torch.cat(x_cnn_odd, dim=1).transpose(1, 2)  # B x D x L -> B x L x D
        x_cnn_even = torch.cat(x_cnn_even, dim=1).transpose(1, 2)  # B x D x L -> B x L x D

        x_gru_odd, _ = self.gru_odd(x_cnn_odd)
        x_gru_even, _ = self.gru_even(x_cnn_even)
        x_gru_odd = self.gru_norm_odd(x_gru_odd)
        x_gru_even = self.gru_norm_even(x_gru_even)

        x_cnn_attention_odd = self.cnn_attention_odd(x_cnn_odd)
        x_cnn_attention_even = self.cnn_attention_even(x_cnn_even)
        x_gru_attention_odd = self.gru_attention_odd(x_gru_odd)
        x_gru_attention_even = self.gru_attention_even(x_gru_even)

        x_avg_pool_odd = torch.mean(x_gru_odd, 1)
        x_max_pool_odd, _ = torch.max(x_gru_odd, 1)
        x_avg_pool_even = torch.mean(x_gru_even, 1)
        x_max_pool_even, _ = torch.max(x_gru_even, 1)

        fm_first = [
            x_cnn_attention_odd,
            x_cnn_attention_even,
            x_gru_attention_odd,
            x_gru_attention_even,
            x_avg_pool_odd,
            x_max_pool_odd,
            x_avg_pool_even,
            x_max_pool_even
        ]

        fm_first = [drop(x) for x, drop in zip(fm_first, self.fm_dropout_layers)]

        fm_second = []
        for t_1, t_2 in itertools.combinations(fm_first, 2):
            fm_second.append(t_1 * t_2)

        x_fc = self.fc(torch.cat(fm_first + fm_second, 1))
        x_fc = self.dropout(self.relu(x_fc))
        outputs = self.output_layer(x_fc)
        return outputs
