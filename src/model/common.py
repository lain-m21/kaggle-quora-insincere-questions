import torch
import torch.nn as nn


class Dense(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(Dense, self).__init__()

        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = self.fc(inputs)
        x = self.relu(x)
        outputs = self.dropout(x)
        return outputs


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True):
        super(Attention, self).__init__()

        self.feature_dim = feature_dim
        self.step_dim = step_dim
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
            a = a * mask.unsqeeze(-1)

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_inputs = inputs * torch.unsqueeze(a, -1)
        return torch.sum(weighted_inputs, 1)

