import math
import torch
from torch_geometric.data import Data
from src.net.layers import (Block, BlockHLA4, BlockHLA8, BlockHLA16,
                            BlockHLA32, BlockHLA64, GCNLayer)


class GCNDHLA(torch.nn.Module):
    def __init__(
            self,
            n_features,
            n_hidden,
            n_classes,
            n_layers,
            dropout,
            norm='None'
            ):
        super().__init__()
        layer = GCNLayer
        n_layers = 2**n_layers

        if n_layers == 2:
            self.block = Block(layer,
                               in_dim=n_features,
                               out_dim=n_hidden,
                               dropout=dropout,
                               norm=norm,
                               out=False)
        elif n_layers == 4:
            self.block = BlockHLA4(layer,
                                   in_dim=n_features,
                                   out_dim=n_hidden,
                                   dropout=dropout,
                                   norm=norm,
                                   out=False)
        elif n_layers == 8:
            self.block = BlockHLA8(layer,
                                   in_dim=n_features,
                                   out_dim=n_hidden,
                                   dropout=dropout,
                                   norm=norm,
                                   out=False)
        elif n_layers == 16:
            self.block = BlockHLA16(layer,
                                    in_dim=n_features,
                                    out_dim=n_hidden,
                                    dropout=dropout,
                                    norm=norm,
                                    out=False)
        elif n_layers == 32:
            self.block = BlockHLA32(layer,
                                    in_dim=n_features,
                                    out_dim=n_hidden,
                                    dropout=dropout,
                                    norm=norm,
                                    out=False)
        elif n_layers == 64:
            self.block = BlockHLA64(layer,
                                    in_dim=n_features,
                                    out_dim=n_hidden,
                                    dropout=dropout,
                                    norm=norm,
                                    out=False)
        self.lin = torch.nn.Linear(n_hidden, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.block.reset_parameters()
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, data:Data):
        x, edge_index = data.x, data.edge_index
        x_g = self.block(x, edge_index)
        out = self.lin(x_g)
        return out

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)