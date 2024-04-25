import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data

from src.net.layers.prop import Prop


class DAGNN(torch.nn.Module):
    def __init__(
        self,
        n_features,
        n_hidden,
        n_classes,
        n_layers,
        dropout,
        k,
        **kwargs    
        ):
        super(DAGNN, self).__init__()


        # TODO: Add more layers
        self.lins = nn.ModuleList()
        self.lin1 = Linear(n_features, n_hidden, **kwargs)
        for _ in range(n_layers - 3): # -3 because we already have 3 layers
            self.lins.append(Linear(in_features=n_hidden, out_features=n_hidden, **kwargs))
        self.lins.append(Linear(in_features=n_hidden, out_features=n_classes, **kwargs))
        self.gc1 = Prop(n_classes, K=k)
        self.dropout = dropout


    def reset_parameters(self):
        self.lin1.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        self.gc1.reset_parameters()


    def forward(self, data:Data):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, nfeat, nnode].
        adj : torch.Tensor
            Adjacency matrix of shape [batch_size, nnode, nnode].
        """

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.lin1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        for lin in self.lins:
            x = F.relu(lin(x))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc1(x, edge_index, norm=edge_attr)

        return x
