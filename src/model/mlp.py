import torch
import torch.nn as nn


class SCDVMLP(nn.Module):
    def __init__(self, input_size, hidden_dims=(256, 64, 64), activation='relu', batchnorm=False,
                 hidden_drop=0.3, out_drop=0.2):
        super(SCDVMLP, self).__init__()

        layers = []
        for i, dim in enumerate(hidden_dims):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_dims[i-1]
            layers.append(nn.Linear(input_dim, dim))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'prelu':
                layers.append(nn.PReLU())
            else:
                layers.append(nn.ReLU())

            if batchnorm:
                layers.append(nn.BatchNorm1d(dim))

            if i == len(hidden_dims) - 1:
                dropout_rate = out_drop
            else:
                dropout_rate = hidden_drop
            layers.append(nn.Dropout(dropout_rate))

        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, inputs):
        x = None
        for i, layer in self.layers:
            if i == 0:
                x = inputs
            else:
                x = layer(x)

        outputs = self.output_layer(x)
        return outputs
