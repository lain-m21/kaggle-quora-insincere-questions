import torch
import torch.nn as nn
import torch.nn.functional as F


class CapsuleUnit(nn.Module):
    def __init__(self, hidden_size, num_capsule=8, dim_capsule=8, routings=4):
        super(CapsuleUnit, self).__init__()

        self.hidden_size = hidden_size
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, hidden_size, num_capsule * dim_capsule)))

    def forward(self, inputs):
        u_hat = torch.matmul(inputs, self.W)
        seq_len = inputs.size(1)
        u_hat = u_hat.view(-1, seq_len, self.num_capsule, self.dim_capsule)
        u_hat = u_hat.permute(0, 2, 1, 3)

        b = torch.zeros_like(u_hat[:, :, :, 0])
        x = None

        for i in range(self.routings):
            b = b.permute(0, 2, 1)
            c = F.softmax(b, dim=2)
            c = c.permute(0, 2, 1)
            b = b.permute(0, 2, 1)
            x = self.squash(torch.einsum('bij,bijk->bik', [c, u_hat]))
            if i < self.routings - 1:
                b = torch.einsum('bik,bijk->bij', [x, u_hat])

        outputs = x.view(-1, self.num_capsule * self.dim_capsule)

        return outputs

    def squash(self, x, dim=-1):
        s_squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = torch.sqrt(s_squared_norm + 1e-8)
        return x / scale
