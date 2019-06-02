import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .positional_encoding import PositionalEncoding


class NeuralNet(nn.Module):
    def __init__(self, ntokens, nhidden):
        super().__init__()
        sizes = [nhidden, 1000, ntokens]
        layers = [
            ('linear_0', nn.Linear(sizes[0], sizes[1])),
            ('tanh_0', nn.Tanh()),
            ('linear_1', nn.Linear(sizes[1], sizes[2]))
        ]
        self.layer = nn.Sequential(OrderedDict(layers))

    def forward(self, hidden):
        out = self.layer(hidden)
        return out


class DenseDecoder(nn.Module):
    TYPE = 'dense'

    def __init__(self, maxlen, ntokens, nhidden, with_pos_idx=False):
        super().__init__()
        nhidden = nhidden + 1 if with_pos_idx else nhidden
        self.decs = nn.ModuleList([NeuralNet(ntokens, nhidden) for _ in range(maxlen)])
        self.with_pos_idx = with_pos_idx
        self._device = None

    def forward(self, hidden, npos):
        # hidden size: [B, H]
        def get_hidden(hidden, pos):
            if not self.with_pos_idx:
                return hidden
            return torch.cat([hidden, torch.full((hidden.size(0), 1), pos).to(self.device)], 1)

        outs = [dec(get_hidden(hidden, idx)).unsqueeze(1) for idx, dec in enumerate(self.decs) if idx < npos]
        outs = torch.cat(outs, 1)
        return outs

    @property
    def device(self):
        # lazy instant
        if self._device is None:
            is_cuda = next(self.parameters()).is_cuda
            self._device = torch.device('cuda' if is_cuda else 'cpu')
        return self._device


class DensePosDecoder(nn.Module):
    TYPE = 'dense_pos'

    def __init__(self, maxlen, ntokens, nhidden, aggr='sum'):
        super().__init__()
        nn_nhidden = nhidden if aggr == 'sum' else nhidden * 2
        self.decs = nn.ModuleList([NeuralNet(ntokens, nn_nhidden) for _ in range(maxlen)])
        self._pos_enc = PositionalEncoding(dropout=0.1, dim=nhidden, aggr=aggr)

    def forward(self, hidden, npos):
        # hidden size: [B, H]
        hidden = hidden.unsqueeze(0).repeat(npos, 1, 1)  # [L, B, H]
        hidden_pos = self._pos_enc(hidden)  # [L, B, H] if aggr == 'sum' else [L, B, 2H]
        outs = [dec(hidden) for dec, hidden in zip(self.decs, hidden_pos)]  # L x [B, V]
        outs = [torch.unsqueeze(x, 1) for x in outs]
        outs = torch.cat(outs, 1)  # [B, L, V]
        return outs


class ConvDecoder(nn.Module):
    TYPE = 'conv'

    def __init__(self, maxlen, ntokens, nhidden):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=(1, 15), stride=1, padding=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(1, 19), stride=1, padding=1)
        self.fc = nn.Linear(68000 // maxlen, ntokens)
        self.maxlen = maxlen

    def forward(self, hidden, npos):
        # hidden size: [B, H]
        # add C_{in}, H_{in} = 1, 1
        batch_size = hidden.size(0)
        hidden = hidden.view(batch_size, 1, 1, -1)  # [B, 1, 1, H]
        outs = F.relu(self.conv1(hidden))
        outs = F.relu(self.conv2(outs))
        outs = outs.view(batch_size, self.maxlen, -1)  # [B, MaxL, X]
        outs = self.fc(outs)[:, :npos, :]  # [B, L, V]
        outs = outs.contiguous()

        return outs

