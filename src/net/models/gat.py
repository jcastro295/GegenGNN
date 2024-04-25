"""

`models.py`
------------

    This script contains the models for the experiments

"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class ParameterizedGAT(nn.Module):
    def __init__(
            self,
            n_features,
            n_hidden,
            n_classes,
            n_layers,
            n_heads,
            dropout,
            **kwargs
        ):
        super(ParameterizedGAT, self).__init__()
        """
        Parameters
        ----------
        nfeat : int
            Number of input features.
        nhid : int
            Number of hidden units.
        nclass : int
            Number of output classes.
        dropout : float
            Dropout rate.
        """

        self.convs = nn.ModuleList()
        if n_layers > 1:
            self.convs.append(GATConv(in_channels=n_features, out_channels=n_hidden,
                                      heads=n_heads, **kwargs))
        for _ in range(n_layers - 2): # -2 because we already have 2 layers
            self.convs.append(GATConv(in_channels=n_hidden, out_channels=n_hidden,
                                      heads=n_heads, **kwargs))
        if n_layers > 1:
            self.convs.append(GATConv(in_channels=n_hidden, out_channels=n_classes,
                                      heads=n_heads, **kwargs))
        else:
            self.convs.append(GATConv(in_channels=n_features, out_channels=n_classes,
                                      heads=n_heads, **kwargs))
        self.dropout = dropout

    def forward(self, data:Data):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, nfeat, nnode].
        adj : torch.Tensor
            Adjacency matrix of shape [batch_size, nnode, nnode].
        """

        x, edge_index = data.x, data.edge_index

        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        #x = F.relu(self.convs[-1](x, adj))

        return x