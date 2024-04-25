"""

`models.py`
------------

    This script contains the models for the experiments

"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class ParameterizedGCN(nn.Module):
    """
    Graph Convolutional Neural Network
    
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
    
    Attributes
    ----------
    gc1 : GraphConvolution
        Graph convolutional layer 1.
    gc2 : GraphConvolution
        Graph convolutional layer 2.
    dropout : float
        Dropout rate.
    
    Methods
    -------
    forward(x, adj)
        Forward pass of the model.
    """

    def __init__(
            self,
            n_features,
            n_hidden,
            n_classes,
            n_layers,
            dropout,
            **kwargs
        ):
        super(ParameterizedGCN, self).__init__()
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

        # TODO: Add more layers
        self.convs = nn.ModuleList()
        self.gc1 = GCNConv(in_channels=n_features, out_channels=n_hidden, **kwargs)
        for _ in range(n_layers - 2): # -2 because we already have 2 layers
            self.convs.append(GCNConv(in_channels=n_hidden, out_channels=n_hidden, **kwargs))
        self.gc2 = GCNConv(in_channels=n_hidden, out_channels=n_classes)
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
        
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        # x = F.relu(self.gc2(x, edge_index))

        return x