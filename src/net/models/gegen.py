"""

`models.py`
------------

    This script contains the models for the experiments

"""

import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.data import Data

from src.net.layers.cascade_layer import CascadeLayer
from src.net.layers.linear_combination import LinearCombinationLayer


class CascadedGeg(nn.Module):
    def __init__(
            self, 
            n_features,
            n_hidden_conv,
            n_hidden_linear,
            n_classes,
            n_conv_layers,
            n_linear_layers,
            filter_order,
            alpha, dropout,
            **kwargs
        ):
        super(CascadedGeg, self).__init__()
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
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if n_conv_layers > 1:
            self.convs.append(
                CascadeLayer(n_features, n_hidden_conv, filter_order,
                                alpha, self.dropout, **kwargs))
        for _ in range(n_conv_layers - 2): # -2 because we already have 2 layers
            self.convs.append(
                CascadeLayer(n_hidden_conv, n_hidden_conv, filter_order, 
                                alpha, self.dropout, **kwargs))
        if n_linear_layers == 0:
            if n_conv_layers > 1:
                input_, output_ = n_hidden_conv, n_classes
            else:
                input_, output_ = n_features, n_classes
        else:
            if n_conv_layers > 1:
                input_, output_ = n_hidden_conv, n_hidden_conv
            else:
                input_, output_ = n_features, n_hidden_conv
        self.convs.append(
                    CascadeLayer(input_, output_, filter_order,
                                    alpha, self.dropout, **kwargs))

        self.linear_combination_layers = nn.ModuleList()

        for _ in range(n_conv_layers):
            self.linear_combination_layers.append(LinearCombinationLayer(filter_order))

        self.lins = nn.ModuleList()
        if n_linear_layers == 1:
            self.lins.append(
                Linear(n_hidden_conv, n_classes, bias=False, weight_initializer='glorot'))
        if n_linear_layers > 1:
            self.lins.append(
                Linear(n_hidden_conv, n_hidden_linear, bias=False, weight_initializer='glorot'))
        for _ in range(n_linear_layers-2): # -2 because we already have the output layer
            self.lins.append(
                Linear(n_hidden_linear, n_hidden_linear, bias=False, weight_initializer='glorot'))
        if n_linear_layers > 1:
            self.lins.append(
                Linear(n_hidden_linear, n_classes, bias=False, weight_initializer='glorot'))

    def forward(self, data: Data):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, nfeat, nnode].
        adj : torch.Tensor
            Adjacency matrix of shape [batch_size, nnode, nnode].
        """

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for i, convs in enumerate(self.convs):
            hs = convs(x, edge_index, edge_weight)
            x = self.linear_combination_layers[i](hs)

        if len(self.lins) > 0:
            for lin in self.lins[:-1]:
                x = lin(x)
                # x = F.elu(x)
                # x = F.dropout(x, self.dropout, training=self.training)
            return self.lins[-1](x)
        else:
            return x
