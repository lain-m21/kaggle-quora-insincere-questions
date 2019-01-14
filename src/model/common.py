import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, masking=True):
        super(Attention, self).__init__()

        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.supports_masking = masking
        self.bias = bias

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, inputs, mask=None):
        eij = torch.mm(
            inputs.contiguous().view(-1, self.feature_dim),
            self.weight
        ).view(-1, self.step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_inputs = inputs * torch.unsqueeze(a, -1)
        return torch.sum(weighted_inputs, 1)


class MultiheadAttention(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout_rate):
        super(MultiheadAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dropout_rate = dropout_rate

        self.q_dense_layer = nn.Linear(_, hidden_dim, bias=False)
        self.k_dense_layer = nn.Linear(_, hidden_dim, bias=False)
        self.v_dense_layer = nn.Linear(_, hidden_dim, bias=False)
        self.output_layer = nn.Linear(_, hidden_dim, bias=False)
        self.attention_dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, memory, mask):
        q = self.q_dense_layer(inputs)  # [batch_size, q_length, hidden_dim]
        k = self.k_dense_layer(memory)  # [batch_size, m_length, hidden_dim]
        v = self.v_dense_layer(memory)  # [batch_size, m_length, hidden_dim]
