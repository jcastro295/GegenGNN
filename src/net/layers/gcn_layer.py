import math

import torch
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter_add

from src.utils.normalization import NeighborNorm


class GCNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, norm='None'):
        super().__init__()
        self.norm = norm

        self.linear = torch.nn.Linear(in_dim, out_dim)

        assert self.norm in ['neighbornorm', 'None']
        if self.norm == 'neighbornorm':
            self.normlayer = NeighborNorm(in_dim)
        else:
            self.normlayer = torch.nn.BatchNorm1d(out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.linear.weight)
        zeros(self.linear.bias)
        if self.norm == 'neighbornorm':
            self.normlayer.reset_parameters()

    def forward(self, x, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index)

        row, col = edge_index
        deg = degree(row)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        if self.norm == 'neighbornorm':
            x_j = self.normlayer(x, edge_index)
        else:
            x_j = x[col]

        x_j = norm.view(-1, 1) * x_j
        out = scatter_add(src=x_j, index=row, dim=0, dim_size=x.size(0))

        if self.norm == 'neighbornorm':
            out = F.relu(self.linear(out))
        else:
            out = self.normlayer(F.relu(self.linear(out)))

        return out

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
