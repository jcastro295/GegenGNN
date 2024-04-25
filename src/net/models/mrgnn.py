import torch
import torch.nn as nn
from torch.functional import F
from torch_geometric.data import Data
from math import floor

from src.net.layers.xhi_layer import XhiLayer


class MRGNN(torch.nn.Module):
    def __init__(
            self,
            n_features,
            n_hidden,
            n_classes,
            n_layers,
            dropout,
            k,
            output=None,
            device='cpu',
            **kwargs
            ):
        super(MRGNN, self).__init__()
        output_list = ['funnel', 'one_layer', 'restricted_funnel']
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.output = output_list[output]
        self.k = k
        self.dropout = torch.nn.Dropout(p=dropout)

        #xhi_layer_mask
        self.convs = nn.ModuleList()

        self.gc1 = XhiLayer(in_channels=n_features, out_channels=n_hidden, k=k, device=device, **kwargs)
        for _ in range(n_layers - 2): # -2 because we already have at least 2 layers
            self.convs.append(XhiLayer(in_channels=n_hidden*k, out_channels=n_hidden, k=k, device=device, **kwargs))

        self.bn_hidden_rec = torch.nn.BatchNorm1d(self.n_hidden*k)
        self.bn_out = torch.nn.BatchNorm1d(self.n_hidden*k)

        self.lin1 = torch.nn.Linear(self.n_hidden*k, self.n_hidden*k)
        self.lin2 = torch.nn.Linear(self.n_hidden*k, floor(self.n_hidden/2)*k)
        self.lin3 = torch.nn.Linear(floor(self.n_hidden/2)*k, self.n_classes)

        if output == "one_layer":
            self.lin1 = torch.nn.Linear(self.n_hidden*k, self.n_classes)

        elif output == "restricted_funnel":
            self.lin1 = torch.nn.Linear(self.n_hidden*k, floor(self.n_hidden/2)*k)
            self.lin2 = torch.nn.Linear(floor(self.n_hidden/2)*k, self.n_classes)

        self.reset_parameters()

    def get_xhi_layer_mask(self, in_channels, out_channels):

        xhi_layer_mask = []
        for i in range(self.k):
            mask_ones = torch.ones(out_channels, in_channels*(i+1))
            mask_zeros = torch.zeros(out_channels, in_channels*(self.k-(i+1)))
            xhi_layer_mask.append(torch.cat([mask_ones, mask_zeros], dim=1))

        return torch.cat(xhi_layer_mask, dim=0)

    def reset_parameters(self):

        self.bn_hidden_rec.reset_parameters()
        self.bn_out.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data:Data):
        x, edge_index = data.x, data.edge_index

        x = self.gc1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)

        if self.output == "funnel" or self.output == "none":
            return self.funnel_output(x)
        elif self.output == "one_layer":
            return self.one_layer_out(x)
        elif self.output == "restricted_funnel":
            return self.restricted_funnel_output(x)
        else:
            assert False, "error in output stage"

    def one_layer_out(self, x):

        x = self.bn_out(x)
        x = self.lin1(x)

        return x

    def funnel_output(self, x):

        x = self.bn_out(x)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = F.relu(self.lin2(x))
        x = self.dropout(x)
        x = self.lin3(x)

        return x

    def restricted_funnel_output(self, x):

        x = self.bn_out(x)
        x = self.dropout(x)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)

        return x