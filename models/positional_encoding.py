import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000, aggr='sum'):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super().__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        self.aggr = aggr

    def forward(self, emb):
        if self.aggr == 'sum':
            emb = emb * math.sqrt(self.dim)
            emb = emb + self.pe[:emb.size(0)]
            emb = self.dropout(emb)
        elif self.aggr == 'cat':
            emb = torch.cat([emb, self.pe[:emb.size(0)].repeat(1, emb.size(1), 1)], dim=-1)
        else:
            raise RuntimeError("Wrong aggregation: {}".format(self.aggr))
        return emb

