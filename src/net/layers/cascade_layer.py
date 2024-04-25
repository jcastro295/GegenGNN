"""

`layers.py`
------------

    This script contains the layers used in the model

"""

import torch.nn as nn
import torch.nn.functional as F
from src.net.layers.gegen_conv import GegConv


class CascadeLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            filter_order,
            alpha,
            dropout,
            **kwargs
        ):
        super(CascadeLayer, self).__init__()

        self.convs = nn.ModuleList()
        for i in range(filter_order):
            self.convs.append(GegConv(in_channels, out_channels, K=i+1, alpha=alpha, **kwargs))

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        hs = []
        for _, conv in enumerate(self.convs):
            h = conv(x, edge_index, edge_weight=edge_weight)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
        return hs
