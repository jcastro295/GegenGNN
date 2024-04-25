import math

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool


class Block(torch.nn.Module):
    def __init__(self,
                 layer,
                 in_dim,
                 out_dim,
                 dropout,
                 num_layers=2,
                 norm='neighbornorm',
                 out=False):
        super().__init__()
        self.dropout = dropout
        self.out = out
        self.conv1 = layer(in_dim=in_dim, out_dim=out_dim, norm='None')

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(layer(in_dim=out_dim, out_dim=out_dim,
                                    norm=norm))

        self.lin = torch.nn.Linear(num_layers * out_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        xd = x
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)
            xs += [x]
        x = torch.cat(xs, dim=-1)
        if self.out:
            x = global_add_pool(x, None)
        x = F.relu(self.lin(x))
        return x


class BlockHLA4(torch.nn.Module):
    def __init__(self,
                 layer,
                 in_dim,
                 out_dim,
                 dropout,
                 num_blocks=2,
                 norm='neighbornorm',
                 out=False):
        super().__init__()
        self.out = out

        self.block1 = Block(layer=layer,
                            in_dim=in_dim,
                            out_dim=out_dim,
                            dropout=dropout,
                            num_layers=num_blocks,
                            norm=norm,
                            out=False)
        self.block2 = Block(layer=layer,
                            in_dim=out_dim,
                            out_dim=out_dim,
                            dropout=dropout,
                            num_layers=num_blocks,
                            norm=norm,
                            out=False)

        self.lin = torch.nn.Linear(num_blocks * out_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.block1.reset_parameters()
        self.block2.reset_parameters()
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = self.block1(x, edge_index)
        xs = [x]
        x = self.block2(x, edge_index)
        xs += [x]

        x = torch.cat(xs, dim=-1)
        if self.out:
            x = global_add_pool(x, None)
        x = F.relu(self.lin(x))
        return x


class BlockHLA8(torch.nn.Module):
    def __init__(self,
                 layer,
                 in_dim,
                 out_dim,
                 dropout,
                 num_blocks=2,
                 norm='neighbornorm',
                 out=False):
        super().__init__()
        self.out = out

        self.block1 = BlockHLA4(layer=layer,
                                in_dim=in_dim,
                                out_dim=out_dim,
                                dropout=dropout,
                                num_blocks=num_blocks,
                                norm=norm,
                                out=False)
        self.block2 = BlockHLA4(layer=layer,
                                in_dim=out_dim,
                                out_dim=out_dim,
                                dropout=dropout,
                                num_blocks=num_blocks,
                                norm=norm,
                                out=False)

        self.lin = torch.nn.Linear(num_blocks * out_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.block1.reset_parameters()
        self.block2.reset_parameters()
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = self.block1(x, edge_index)
        xs = [x]

        x = self.block2(x, edge_index)
        xs += [x]

        x = torch.cat(xs, dim=-1)
        if self.out:
            x = global_add_pool(x, None)
        x = F.relu(self.lin(x))
        return x


class BlockHLA16(torch.nn.Module):
    def __init__(self,
                 layer,
                 in_dim,
                 out_dim,
                 dropout,
                 num_blocks=2,
                 norm='neighbornorm',
                 out=False):
        super().__init__()
        self.out = out

        self.block1 = BlockHLA8(layer=layer,
                                in_dim=in_dim,
                                out_dim=out_dim,
                                dropout=dropout,
                                num_blocks=num_blocks,
                                norm=norm,
                                out=False)
        self.block2 = BlockHLA8(layer=layer,
                                in_dim=out_dim,
                                out_dim=out_dim,
                                dropout=dropout,
                                num_blocks=num_blocks,
                                norm=norm,
                                out=False)

        self.lin = torch.nn.Linear(num_blocks * out_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.block1.reset_parameters()
        self.block2.reset_parameters()
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = self.block1(x, edge_index)
        xs = [x]

        x = self.block2(x, edge_index)
        xs += [x]

        x = torch.cat(xs, dim=-1)
        if self.out:
            x = global_add_pool(x, None)
        x = F.relu(self.lin(x))
        return x


class BlockHLA32(torch.nn.Module):
    def __init__(self,
                 layer,
                 in_dim,
                 out_dim,
                 dropout,
                 num_blocks=2,
                 norm='neighbornorm',
                 out=False):
        super().__init__()
        self.out = out

        self.block1 = BlockHLA16(layer=layer,
                                 in_dim=in_dim,
                                 out_dim=out_dim,
                                 dropout=dropout,
                                 num_blocks=num_blocks,
                                 norm=norm,
                                 out=False)
        self.block2 = BlockHLA16(layer=layer,
                                 in_dim=out_dim,
                                 out_dim=out_dim,
                                 dropout=dropout,
                                 num_blocks=num_blocks,
                                 norm=norm,
                                 out=False)

        self.lin = torch.nn.Linear(num_blocks * out_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.block1.reset_parameters()
        self.block2.reset_parameters()
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = self.block1(x, edge_index)
        xs = [x]

        x = self.block2(x, edge_index)
        xs += [x]

        x = torch.cat(xs, dim=-1)
        if self.out:
            x = global_add_pool(x, None)
        x = F.relu(self.lin(x))
        return x


class BlockHLA64(torch.nn.Module):
    def __init__(self,
                 layer,
                 in_dim,
                 out_dim,
                 dropout,
                 num_blocks=2,
                 norm='neighbornorm',
                 out=False):
        super().__init__()
        self.out = out

        self.block1 = BlockHLA32(layer=layer,
                                 in_dim=in_dim,
                                 out_dim=out_dim,
                                 dropout=dropout,
                                 num_blocks=num_blocks,
                                 norm=norm)
        self.block2 = BlockHLA32(layer=layer,
                                 in_dim=out_dim,
                                 out_dim=out_dim,
                                 dropout=dropout,
                                 num_blocks=num_blocks,
                                 norm=norm)

        self.lin = torch.nn.Linear(num_blocks * out_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.block1.reset_parameters()
        self.block2.reset_parameters()
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = self.block1(x, edge_index)
        xs = [x]

        x = self.block2(x, edge_index)
        xs += [x]

        x = torch.cat(xs, dim=-1)
        if self.out:
            x = global_add_pool(x, None)
        x = F.relu(self.lin(x))
        return x

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
