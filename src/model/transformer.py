import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StableSoftMax(nn.Module):
    def __init__(self):
        super(StableSoftMax, self).__init__()

    def forward(self, inputs):
        x = torch.exp(inputs)
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-10)
        return x


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = StableSoftMax()
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    """

    def __init__(self, num_head, embed_dim, k_dim, v_dim, dropout=0.1):
        super().__init__()

        self.num_head = num_head
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.w_qs = nn.Linear(embed_dim, num_head * k_dim)
        self.w_ks = nn.Linear(embed_dim, num_head * k_dim)
        self.w_vs = nn.Linear(embed_dim, num_head * v_dim)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (embed_dim + k_dim)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (embed_dim + k_dim)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (embed_dim + v_dim)))

        self.attention = ScaledDotProductAttention(temperature=np.power(k_dim, 0.5))
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.fc = nn.Linear(num_head * v_dim, embed_dim)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.k_dim, self.v_dim, self.num_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n * b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n * b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n * b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n * b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """
    A two-feed-forward-layer module
    """

    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(in_dim, hidden_dim, 1)  # position-wise
        self.w_2 = nn.Conv1d(hidden_dim, in_dim, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderUnitLayer(nn.Module):
    """
    Transformer's Encoder unit layer in the Encoder
    """
    def __init__(self, embed_dim, inner_dim, num_head, k_dim, v_dim, dropout=0.1):
        super(EncoderUnitLayer, self).__init__()
        self.self_attention = MultiHeadAttention(num_head, embed_dim, k_dim, v_dim, dropout=dropout)
        # self.feedforward = PositionwiseFeedForward(embed_dim, inner_dim, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.self_attention(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        # enc_output = self.feedforward(enc_output)
        # enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


def get_sinusoid_encoding_table(n_position, hidden_dim, padding_idx=None):
    """
    Sinusoid position encoding table
    """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / hidden_dim)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(hidden_dim)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """
    For masking out the padding part of key sequence.
    """

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class TransformerEncoder(nn.Module):
    """
    An encoder model with self attention mechanism for transformer.
    """

    def __init__(self,
                 embedding_matrix, seq_length, out_hidden_dim=64, out_drop=0.5,
                 num_layers=2, num_head=8, k_dim=16, v_dim=16, inner_dim=256,
                 dropout=0.3):
        super().__init__()

        n_position = seq_length + 1
        embed_dim = embedding_matrix.shape[1]

        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)

        self.position_encoder = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, embed_dim, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderUnitLayer(embed_dim, inner_dim, num_head, k_dim, v_dim, dropout=dropout)
            for _ in range(num_layers)])

        self.fc = nn.Linear(embed_dim * 2, out_hidden_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(out_drop)
        self.output_layer = nn.Linear(out_hidden_dim, 1)

    def forward(self, inputs):
        src_seq, src_pos = inputs['sequence'], inputs['position']

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.word_embedding(src_seq) + self.position_encoder(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)

        pool_avg = torch.mean(enc_output, 1)
        pool_max, _ = torch.max(enc_output, 1)

        x_fc = torch.cat([pool_avg, pool_max], dim=1)
        x_fc = self.drop(self.relu(self.fc(x_fc)))
        outputs = self.output_layer(x_fc)

        return outputs
